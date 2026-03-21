import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "BridgeCompare — Cross-Chain Bridge Fee Comparison",
  description: "Compare fees and speeds across Across, Stargate, CCIP, and CCTP bridge protocols in real-time.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <nav className="top-nav">
          <Link href="/" className="nav-link">Bridge Compare</Link>
          <Link href="/data" className="nav-link">Training Data</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
