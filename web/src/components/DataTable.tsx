import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";

export type Col<T> = ColumnDef<T, any> & { meta?: { align?: "r"; wrap?: boolean } };

/** Server-paged table: the API does filtering and paging; this only renders. */
export function DataTable<T>({ data, columns, onRowClick, rowKey }: {
  data: T[];
  columns: Col<T>[];
  onRowClick?: (row: T) => void;
  rowKey: (row: T) => string;
}) {
  const table = useReactTable({
    data, columns, getCoreRowModel: getCoreRowModel(), manualPagination: true,
    getRowId: (r) => rowKey(r),
  });
  return (
    <div className="table-wrap">
      <table>
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => {
                const meta = (h.column.columnDef as Col<T>).meta;
                return <th key={h.id} className={meta?.align ?? ""}>{flexRender(h.column.columnDef.header, h.getContext())}</th>;
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className={onRowClick ? "click" : ""}
              onClick={onRowClick ? () => onRowClick(row.original) : undefined}
              onKeyDown={onRowClick ? (e) => { if (e.key === "Enter") onRowClick(row.original); } : undefined}
              tabIndex={onRowClick ? 0 : undefined}>
              {row.getVisibleCells().map((cell) => {
                const meta = (cell.column.columnDef as Col<T>).meta;
                return <td key={cell.id} className={`${meta?.align ?? ""} ${meta?.wrap ? "wrap" : ""}`}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
