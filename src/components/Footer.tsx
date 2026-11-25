import React from 'react';
import { BrandLogo } from './BrandLogo';
import { WORKBENCH_URL } from './Navbar';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-white dark:bg-[#16181D] border-t border-[#E6E5DF] dark:border-[#2B2E36] transition-colors duration-200">
      <div className="mx-auto max-w-[1220px] px-5 sm:px-8 py-12">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <BrandLogo compact />
              <span className="text-[14px] font-semibold text-[#141416] dark:text-[#F3F4F6]">
                Ground Truth (GT-Transformer &amp; recombx)
              </span>
            </div>
            <p className="text-[13px] text-[#5A5A62] dark:text-[#A8ADB8]">
              Authors: Smyan Reddy &amp; Darren P. Martin <span aria-hidden="true">·</span> University of Cape Town
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-6 text-[13px]">
            <a
              href="#overview"
              className="text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors"
            >
              Overview
            </a>
            <a
              href="#explorer-3d"
              className="text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors"
            >
              3D Helix
            </a>
            <a
              href="#walkthrough"
              className="text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors"
            >
              Read Scan
            </a>
            <a
              href="#benchmarks"
              className="text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors"
            >
              Benchmarks
            </a>
            <a
              href="#pipeline"
              className="text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors"
            >
              Pipeline
            </a>
            <a
              href="#access"
              className="text-[#5A5A62] dark:text-[#A8ADB8] hover:text-[#141416] dark:hover:text-[#F3F4F6] transition-colors"
            >
              CLI &amp; Desktop
            </a>
            <a
              href={WORKBENCH_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-[#4F6BF6] dark:text-[#2DD4A4] hover:text-[#3A53D6] dark:hover:text-[#5EEAD4] transition-colors whitespace-nowrap"
            >
              Web Workbench →
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};
