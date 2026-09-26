import { useEffect, useState } from "react";

// Hash routing: the console is served as static files by the API process, so
// hash URLs work on any path and survive a reload without server rewrites.
export type Route = { page: string; id: string | null };

function parse(): Route {
  const [page = "overview", id = null] = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  return { page: page || "overview", id: id || null };
}

export function useRoute(): Route {
  const [r, setR] = useState(parse);
  useEffect(() => {
    const on = () => setR(parse());
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return r;
}

export const href = (page: string, id?: string | null) =>
  `#/${page}${id ? `/${encodeURIComponent(id)}` : ""}`;

export const go = (page: string, id?: string | null) => { location.hash = href(page, id); };
