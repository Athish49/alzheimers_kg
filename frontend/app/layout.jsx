import '@/styles/globals.css';
import '@/styles/app.css';
import '@/styles/home.css';

export const metadata = {
  title: {
    default: 'Atlas — Alzheimer\'s Knowledge Graph',
    template: '%s | Atlas',
  },
  description:
    'An ontology-grounded Graph RAG interface for Alzheimer\'s disease research. Ask questions in plain English; every answer traces to a node in a curated biomedical knowledge graph.',
  keywords: [
    "Alzheimer's disease",
    'knowledge graph',
    'biomarker',
    'graph RAG',
    'neo4j',
    'ontology',
    'drug trials',
    'research tool',
  ],
  authors: [{ name: 'Athish Gopal Rajesh', url: 'https://athish-gopal-rajesh.vercel.app/' }],
  creator: 'Athish Gopal Rajesh',
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL || 'https://atlas-alzheimers.vercel.app'
  ),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: '/',
    siteName: 'Atlas',
    title: "Atlas — Alzheimer's Knowledge Graph",
    description:
      "Ontology-grounded Graph RAG interface for Alzheimer's disease. Every answer traces to a node.",
  },
  twitter: {
    card: 'summary_large_image',
    title: "Atlas — Alzheimer's Knowledge Graph",
    description:
      "Ontology-grounded Graph RAG interface for Alzheimer's disease. Every answer traces to a node.",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true },
  },
  alternates: {
    canonical: '/',
  },
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: '16x16', type: 'image/x-icon' },
      { url: '/favicon.svg', type: 'image/svg+xml' },
    ],
  },
  manifest: '/site.webmanifest',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <a className="skip-link" href="#main-content">Skip to main content</a>
        {children}
      </body>
    </html>
  );
}
