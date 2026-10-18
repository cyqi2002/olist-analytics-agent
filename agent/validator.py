"""Static and dry-run checks on generated SQL before it touches the data.

1. parse with sqlglot (DuckDB dialect) - must be exactly one statement
2. read-only: the statement must be a SELECT (or UNION/CTE of SELECTs) with no DDL/DML anywhere
3. every referenced table and column must exist
4. add a LIMIT if the outermost query has none
5. EXPLAIN the query so DuckDB's binder catches type and function errors without running it
"""

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from agent.config import ROW_LIMIT
from agent.db import catalog, cursor

FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Pragma, exp.Copy, exp.Attach, exp.Detach, exp.Set, exp.Use, exp.Transaction,
)
# table functions that read files or remote URLs
FORBIDDEN_FUNCTIONS = {"read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
                       "read_text", "read_blob", "glob", "query_table", "query"}


@dataclass
class ValidationResult:
    ok: bool
    sql: str  # possibly rewritten (LIMIT added)
    errors: list[str] = field(default_factory=list)
    limit_added: bool = False


def _fail(sql: str, msg: str) -> ValidationResult:
    return ValidationResult(ok=False, sql=sql, errors=[msg])


def _check_references(tree: exp.Expression) -> list[str]:
    cat = catalog()
    errors = []
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    # map alias -> real table for every table reference
    alias_to_table: dict[str, str] = {}
    real_tables: set[str] = set()
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if not name:
            continue  # table function such as unnest(...)
        if name in cte_names:
            continue
        if name not in cat:
            errors.append(f"unknown table '{t.name}'. Available tables: {', '.join(sorted(cat))}")
            continue
        real_tables.add(name)
        alias_to_table[(t.alias or name).lower()] = name

    # columns produced by the query itself (select aliases, CTE / subquery outputs) are fine
    derived = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    for node in tree.find_all(exp.TableAlias):
        derived.update(c.name.lower() for c in node.columns)
        derived.add(node.name.lower())
    for sub in tree.find_all(exp.Subquery, exp.CTE):
        if isinstance(sub.this, exp.Select):
            derived.update(s.alias_or_name.lower() for s in sub.this.selects)
    known_columns = {c.lower() for t in real_tables for c in cat[t]}

    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if not name or name == "*":
            continue
        qualifier = col.table.lower()
        if qualifier and qualifier in alias_to_table:
            table = alias_to_table[qualifier]
            if name not in {c.lower() for c in cat[table]}:
                errors.append(f"column '{col.name}' does not exist in table '{table}'. "
                              f"Columns: {', '.join(cat[table])}")
        elif not qualifier and name not in known_columns and name not in derived:
            errors.append(f"column '{col.name}' does not exist in the tables used "
                          f"({', '.join(sorted(real_tables)) or 'none'})")
    return sorted(set(errors))


def validate(sql: str) -> ValidationResult:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return _fail(sql, "empty query")

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        return _fail(sql, f"SQL syntax error: {e}")
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return _fail(sql, f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        return _fail(sql, f"only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN):
            return _fail(sql, f"forbidden operation in query: {node.key.upper()}")
        if isinstance(node, exp.Func) and node.sql_name().lower() in FORBIDDEN_FUNCTIONS:
            return _fail(sql, f"forbidden function: {node.sql_name()}")
        if isinstance(node, exp.Anonymous) and str(node.this).lower() in FORBIDDEN_FUNCTIONS:
            return _fail(sql, f"forbidden function: {node.this}")

    errors = _check_references(tree)
    if errors:
        return ValidationResult(ok=False, sql=sql, errors=errors)

    limit_added = False
    if tree.args.get("limit") is None:
        tree = tree.limit(ROW_LIMIT)
        limit_added = True
    final_sql = tree.sql(dialect="duckdb", pretty=True)

    try:
        cur = cursor()
        cur.execute(f"EXPLAIN {final_sql}")
        cur.close()
    except Exception as e:  # noqa: BLE001 - DuckDB raises several error types here
        return ValidationResult(ok=False, sql=final_sql, errors=[f"{type(e).__name__}: {e}"])

    return ValidationResult(ok=True, sql=final_sql, limit_added=limit_added)
