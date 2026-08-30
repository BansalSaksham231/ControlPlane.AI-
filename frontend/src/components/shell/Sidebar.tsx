"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck, PanelLeftClose, PanelLeft } from "lucide-react";

import { cn } from "@/lib/utils";
import { NAV_ITEMS, type NavItem } from "./nav";

function isActive(pathname: string, item: NavItem): boolean {
  return item.match === "exact"
    ? pathname === item.href
    : pathname === item.href || pathname.startsWith(`${item.href}/`);
}

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  /** called after navigation on mobile so the drawer closes */
  onNavigate?: () => void;
}

export function Sidebar({
  collapsed,
  onToggleCollapse,
  onNavigate,
}: SidebarProps) {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col bg-card">
      <div
        className={cn(
          "flex h-14 items-center border-b border-border px-3",
          collapsed ? "justify-center" : "justify-between",
        )}
      >
        <Link
          href="/dashboard"
          onClick={onNavigate}
          aria-label="ControlPlane.ai home"
          className="flex items-center gap-2 font-semibold"
        >
          <ShieldCheck className="h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
          {!collapsed && <span className="text-sm">ControlPlane.ai</span>}
        </Link>
        {!collapsed && (
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label="Collapse sidebar"
            className="hidden rounded-md p-1.5 text-muted-foreground hover:bg-muted lg:block"
          >
            <PanelLeftClose className="h-4 w-4" aria-hidden="true" />
          </button>
        )}
      </div>

      <nav aria-label="Primary" className="flex-1 p-2">
        {collapsed && (
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label="Expand sidebar"
            className="mb-1 hidden w-full items-center justify-center rounded-md p-2 text-muted-foreground hover:bg-muted lg:flex"
          >
            <PanelLeft className="h-4 w-4" aria-hidden="true" />
          </button>
        )}
        <ul className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = isActive(pathname, item);
            const Icon = item.icon;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  onClick={onNavigate}
                  title={collapsed ? item.label : undefined}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    collapsed && "justify-center px-2",
                    active
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {!collapsed && (
        <div className="border-t border-border p-3 text-[11px] leading-relaxed text-muted-foreground">
          Deterministic, offline risk control plane. Automated decisions are
          immutable; governance is an append-only track.
        </div>
      )}
    </div>
  );
}
