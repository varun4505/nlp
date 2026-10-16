import type { Metadata } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";
import "./globals.css";
import { NarratorProvider } from "@/lib/narrator";
import { Nav, NarratorBar } from "@/components/ui";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const serif = Instrument_Serif({ variable: "--font-instrument", subsets: ["latin"], weight: "400", style: ["normal", "italic"] });

export const metadata: Metadata = {
  title: "DriftLens",
  description: "Detecting moments of change in social-media timelines with classical NLP.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} ${serif.variable} antialiased`}>
        <NarratorProvider>
          <Nav />
          <main className="mx-auto max-w-6xl px-4 pb-32 pt-8">{children}</main>
          <footer className="mx-auto max-w-6xl px-4 pb-28 text-xs text-ink-3">
            DriftLens is not a diagnostic or crisis tool. In India, Tele-MANAS
            offers free 24x7 support at 14416.
          </footer>
          <NarratorBar />
        </NarratorProvider>
      </body>
    </html>
  );
}
