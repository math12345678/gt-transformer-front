import { useEffect, useRef, useState } from 'react';

interface RevealOptions {
  threshold?: number;
  rootMargin?: string;
}

/**
 * Lightweight IntersectionObserver hook for subtle one-shot scroll reveals.
 * Respects prefers-reduced-motion synchronously and degrades safely when
 * IntersectionObserver is unavailable so content is always reachable.
 */
export function useRevealOnScroll<T extends HTMLElement = HTMLDivElement>(
  options: RevealOptions = {}
) {
  const { threshold = 0.08, rootMargin = '0px 0px -36px 0px' } = options;
  const ref = useRef<T | null>(null);
  const [isVisible, setIsVisible] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true;
    if (typeof window.matchMedia === 'function') {
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return true;
      }
    }
    return typeof IntersectionObserver === 'undefined';
  });

  useEffect(() => {
    if (isVisible) return;
    const node = ref.current;
    if (!node || typeof IntersectionObserver === 'undefined') {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setIsVisible(true);
            observer.disconnect();
            break;
          }
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [isVisible, threshold, rootMargin]);

  return { ref, isVisible };
}
