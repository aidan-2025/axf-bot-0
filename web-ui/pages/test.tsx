import React from 'react';
import Head from 'next/head';

export default function TestPage() {
  return (
    <>
      <Head>
        <title>Test Page</title>
      </Head>
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Test Page</h1>
          <p className="text-gray-600">This is a simple test page to verify Next.js is working.</p>
        </div>
      </div>
    </>
  );
}