import { NextApiRequest, NextApiResponse } from 'next';
import os from 'os';

interface HealthCheckResponse {
  status: string;
  timestamp: string;
  service: string;
  version: string;
  checks?: {
    memory_usage: string;
    cpu_usage: string;
    uptime: number;
    node_version: string;
    platform: string;
  };
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<HealthCheckResponse>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      service: 'web-ui',
      version: '1.0.0'
    });
  }

  try {
    // Basic health check
    if (req.query.detailed !== 'true') {
      return res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: 'web-ui',
        version: '1.0.0'
      });
    }

    // Detailed health check
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    const uptime = process.uptime();

    const checks = {
      memory_usage: `${Math.round((memoryUsage.heapUsed / 1024 / 1024) * 100) / 100} MB`,
      cpu_usage: `${Math.round((cpuUsage.user + cpuUsage.system) / 1000000)} ms`,
      uptime: Math.round(uptime),
      node_version: process.version,
      platform: os.platform()
    };

    return res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'web-ui',
      version: '1.0.0',
      checks
    });

  } catch (error) {
    return res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'web-ui',
      version: '1.0.0'
    });
  }
}
