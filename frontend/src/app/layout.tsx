import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";
import { AppShell } from "@/components/shell/AppShell";
import { SWRProvider } from "@/providers/swr-provider";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: {
    default: "ControlPlane.ai",
    template: "%s · ControlPlane.ai",
  },
  description:
    "Enterprise AI risk governance control plane — deterministic, offline decisioning with an immutable audit trail.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <SWRProvider>
          <AppShell>{children}</AppShell>
        </SWRProvider>
      </body>
    </html>
  );
}
