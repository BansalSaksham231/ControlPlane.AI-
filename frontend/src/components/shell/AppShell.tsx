"use client";

import { useCallback, useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

/**
 * Responsive enterprise shell.
 *
 *  - lg and up : persistent sidebar, collapsible to an icon rail
 *  - below lg  : sidebar hidden; the header hamburger opens it as a modal
 *                drawer that closes on navigation, backdrop click or Escape
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [isRailCollapsed, setIsRailCollapsed] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const closeDrawer = useCallback(() => setIsDrawerOpen(false), []);

  // close the mobile drawer whenever the route changes
  useEffect(() => {
    closeDrawer();
  }, [pathname, closeDrawer]);

  // Escape closes the drawer; lock body scroll while it is open
  useEffect(() => {
    if (!isDrawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeDrawer();
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = "";
    };
  }, [isDrawerOpen, closeDrawer]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-sm focus:text-primary-foreground"
      >
        Skip to main content
      </a>

      {/* desktop sidebar */}
      <aside
        aria-label="Sidebar"
        className={cn(
          "fixed inset-y-0 left-0 z-40 hidden border-r border-border transition-[width] duration-200 lg:block",
          isRailCollapsed ? "w-[64px]" : "w-64",
        )}
      >
        <Sidebar
          collapsed={isRailCollapsed}
          onToggleCollapse={() => setIsRailCollapsed((collapsed) => !collapsed)}
        />
      </aside>

      {/* mobile drawer */}
      <div
        className={cn(
          "fixed inset-0 z-50 lg:hidden",
          isDrawerOpen ? "pointer-events-auto" : "pointer-events-none",
        )}
        aria-hidden={isDrawerOpen ? undefined : "true"}
      >
        <div
          className={cn(
            "absolute inset-0 bg-black/50 transition-opacity",
            isDrawerOpen ? "opacity-100" : "opacity-0",
          )}
          onClick={closeDrawer}
        />
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Navigation menu"
          className={cn(
            "absolute inset-y-0 left-0 w-72 max-w-[85vw] shadow-xl transition-transform duration-200",
            isDrawerOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <button
            type="button"
            onClick={closeDrawer}
            aria-label="Close navigation menu"
            className="absolute right-2 top-2 z-10 rounded-md p-2 text-muted-foreground hover:bg-muted"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
          <Sidebar
            collapsed={false}
            onToggleCollapse={() => undefined}
            onNavigate={closeDrawer}
          />
        </div>
      </div>

      {/* main column */}
      <div
        className={cn(
          "flex min-h-screen flex-col transition-[padding] duration-200",
          isRailCollapsed ? "lg:pl-[64px]" : "lg:pl-64",
        )}
      >
        <Header onOpenSidebar={() => setIsDrawerOpen(true)} />
        <main
          id="main-content"
          className="mx-auto w-full max-w-7xl flex-1 p-4 sm:p-6"
        >
          {children}
        </main>
      </div>
    </div>
  );
}
