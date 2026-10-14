import type { ReactNode } from "react";

import { href, type Route } from "../lib/router";
import { BookIcon, DatasetIcon, ExploreIcon, MicIcon, PlusIcon, TrophyIcon } from "./Icons";

const NAV: Array<{ route: Route; label: string; icon: (p: { className?: string }) => ReactNode }> = [
  { route: { page: "explore" }, label: "Explore", icon: ExploreIcon },
  { route: { page: "analyze" }, label: "Analyze", icon: MicIcon },
  { route: { page: "dataset" }, label: "Dataset", icon: DatasetIcon },
  { route: { page: "benchmarks" }, label: "Benchmarks", icon: TrophyIcon },
  { route: { page: "method" }, label: "Method", icon: BookIcon },
];

function Mark() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
      <rect width="28" height="28" rx="7" fill="var(--color-brand)" />
      <path
        d="M5 15c2-5 3-5 4.5 0s2.5 5 4.5 0 2.5-7 4.5 0 2 4 4.5 0"
        fill="none"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function Shell({ current, children }: { current: Route["page"]; children: ReactNode }) {
  return (
    <div className="min-h-screen bg-canvas">
      <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-line bg-canvas/95 px-4 backdrop-blur md:px-6">
        <a href="#/" className="flex items-center gap-2.5 text-[17px] font-semibold tracking-tight text-ink">
          <Mark />
          Oratio
        </a>
        <div className="ml-2 hidden flex-1 md:block">
          <label className="relative block max-w-xl">
            <span className="sr-only">Search samples</span>
            <ExploreIcon className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" />
            <input
              type="search"
              placeholder="Search speeches, speakers and flaws"
              onChange={(event) => {
                window.dispatchEvent(new CustomEvent("oratio:search", { detail: event.target.value }));
                if (window.location.hash !== "#/" && window.location.hash !== "") window.location.hash = "#/";
              }}
              className="h-10 w-full rounded-full border border-line bg-surface pl-10 pr-4 text-sm outline-none transition focus:border-brand focus:bg-canvas"
            />
          </label>
        </div>
        <a
          href="#/analyze"
          className="ml-auto inline-flex h-10 items-center gap-1.5 rounded-full bg-ink px-4 text-sm font-medium text-white hover:bg-brand"
        >
          <PlusIcon className="size-4" />
          New analysis
        </a>
      </header>
      <div className="flex">
        <nav
          aria-label="Main"
          className="sticky top-16 hidden h-[calc(100vh-4rem)] w-56 shrink-0 flex-col gap-1 border-r border-line px-3 py-4 md:flex"
        >
          {NAV.map(({ route, label, icon: Icon }) => {
            const active = route.page === current || (current === "report" && route.page === "explore");
            return (
              <a
                key={label}
                href={href(route)}
                aria-current={active ? "page" : undefined}
                className={`flex h-10 items-center gap-3 rounded-full px-4 text-sm ${
                  active ? "bg-brand-soft font-semibold text-brand" : "text-ink hover:bg-surface"
                }`}
              >
                <Icon />
                {label}
              </a>
            );
          })}
          <p className="mt-auto px-4 text-xs leading-relaxed text-muted">
            Multimodal AI Hackathon 2026, Track C. Open dataset and code.
          </p>
        </nav>
        <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-30 flex border-t border-line bg-canvas md:hidden">
          {NAV.map(({ route, label, icon: Icon }) => (
            <a
              key={label}
              href={href(route)}
              className={`flex flex-1 flex-col items-center gap-0.5 py-2 text-[11px] ${
                route.page === current ? "text-brand" : "text-muted"
              }`}
            >
              <Icon />
              {label}
            </a>
          ))}
        </nav>
        <main className="min-w-0 flex-1 pb-20 md:pb-10">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  aside,
}: {
  title: string;
  subtitle?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 px-4 pt-8 md:px-8">
      <div className="min-w-0">
        <h1 className="text-[28px] font-semibold leading-tight tracking-tight">{title}</h1>
        {subtitle && <div className="mt-1.5 text-[15px] text-muted">{subtitle}</div>}
      </div>
      {aside}
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "error"; children: ReactNode }) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={`rounded-xl border px-4 py-3 text-sm ${
        tone === "error" ? "border-bad/30 bg-bad/5 text-bad" : "border-line bg-surface text-muted"
      }`}
    >
      {children}
    </div>
  );
}
