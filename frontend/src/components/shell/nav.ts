import {
  LayoutDashboard,
  ShieldAlert,
  Radio,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
  /** match on exact href, or any path that starts with it */
  match: "exact" | "prefix";
}

export const NAV_ITEMS: NavItem[] = [
  {
    label: "Command Center",
    href: "/dashboard",
    icon: LayoutDashboard,
    match: "exact",
  },
  {
    label: "Incident Investigation",
    href: "/incident",
    icon: ShieldAlert,
    match: "prefix",
  },
  {
    label: "Live Governance",
    href: "/check",
    icon: Radio,
    match: "prefix",
  },
];
