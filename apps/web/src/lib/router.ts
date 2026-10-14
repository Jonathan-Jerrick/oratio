import { useEffect, useState } from "react";

export type Route =
  | { page: "explore" }
  | { page: "analyze" }
  | { page: "report"; source: "sample" | "upload"; id: string }
  | { page: "dataset" }
  | { page: "benchmarks" }
  | { page: "method" };

export function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  switch (parts[0]) {
    case "analyze":
      return { page: "analyze" };
    case "report":
      return { page: "report", source: parts[1] === "upload" ? "upload" : "sample", id: parts[2] ?? "" };
    case "dataset":
      return { page: "dataset" };
    case "benchmarks":
      return { page: "benchmarks" };
    case "method":
      return { page: "method" };
    default:
      return { page: "explore" };
  }
}

export function href(route: Route): string {
  if (route.page === "report") return `#/report/${route.source}/${encodeURIComponent(route.id)}`;
  return route.page === "explore" ? "#/" : `#/${route.page}`;
}

export function navigate(route: Route): void {
  window.location.hash = href(route);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
