import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Prόvlepsi AI · Time-series AutoML studio",
  description:
    "Privacy-first time-series AutoML in your browser: upload a CSV, explore it, clean it, and train ARIMA, scikit-learn and deep-learning forecasters without your data leaving the device.",
};

// Applies the saved theme before first paint so there is no flash.
const themeScript = `try{var s=JSON.parse(localStorage.getItem("provlepsi-settings")||"{}");document.documentElement.dataset.theme=(s.state&&s.state.theme)||"dark"}catch(e){document.documentElement.dataset.theme="dark"}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen font-sans">{children}</body>
    </html>
  );
}
