import "./globals.css";
export const metadata = { title: "Nexvia Bank — Fraud Protection Prototype", description: "Banking app prototype with real-time fraud detection" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <footer className="footer">Nexvia Bank prototype · Fraud detection demo (IEEE-CIS inspired) · This is a prototype — no real money moves. Full ML backend in <code>/backend</code>.</footer>
      </body>
    </html>
  );
}
