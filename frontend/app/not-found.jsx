import Link from 'next/link';

export const metadata = {
  title: '404 — Page Not Found',
  robots: { index: false },
};

export default function NotFound() {
  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'var(--font-sans)',
      background: 'var(--bg)',
      color: 'var(--fg)',
      gap: 12,
      padding: '0 24px',
      textAlign: 'center',
    }}>
      <div style={{
        width: 48, height: 48, borderRadius: 12,
        background: 'var(--fg)', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        fontFamily: 'var(--font-serif)', fontStyle: 'italic',
        fontSize: 28, color: 'var(--bg)', marginBottom: 8,
      }}>A</div>
      <div style={{ fontSize: 13, letterSpacing: '0.08em', textTransform: 'uppercase', opacity: 0.45 }}>
        404
      </div>
      <h1 style={{ fontSize: 24, fontWeight: 600, margin: '4px 0', lineHeight: 1.3 }}>
        Page not found
      </h1>
      <p style={{ fontSize: 15, color: 'var(--fg-secondary)', maxWidth: 340, lineHeight: 1.6, margin: 0 }}>
        This URL doesn&apos;t exist. Head back to the workbench or the home page.
      </p>
      <div style={{ display: 'flex', gap: 10, marginTop: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
        <Link href="/" style={{
          padding: '9px 18px', background: 'var(--fg)', color: 'var(--bg)',
          borderRadius: 'var(--r-md)', fontSize: 14, fontWeight: 500,
          textDecoration: 'none',
        }}>
          Home
        </Link>
        <Link href="/app" style={{
          padding: '9px 18px', border: '1px solid var(--border-strong)',
          borderRadius: 'var(--r-md)', fontSize: 14, fontWeight: 500,
          textDecoration: 'none', color: 'var(--fg)',
        }}>
          Open Atlas
        </Link>
      </div>
    </div>
  );
}
