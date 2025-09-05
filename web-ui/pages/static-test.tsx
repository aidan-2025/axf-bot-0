import Head from 'next/head';

export default function StaticTestPage() {
  return (
    <>
      <Head>
        <title>Static Test - AXF Bot 0</title>
      </Head>
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Static Test Page</h1>
          <p className="text-xl text-gray-600 mb-8">This page should load immediately without any API calls.</p>
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
            ✅ Static rendering is working
          </div>
          <div className="mt-4">
            <a href="/api-test" className="text-blue-600 hover:text-blue-800 underline">
              Go to API Test Page
            </a>
          </div>
        </div>
      </div>
    </>
  );
}
