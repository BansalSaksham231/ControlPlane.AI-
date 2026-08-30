"use client";

import { SWRConfig } from "swr";

/** App-wide SWR defaults. Individual hooks override as needed. */
export function SWRProvider({ children }: { children: React.ReactNode }) {
  return (
    <SWRConfig
      value={{
        revalidateOnFocus: false,
        shouldRetryOnError: false,
        errorRetryCount: 0,
      }}
    >
      {children}
    </SWRConfig>
  );
}
