import assert from 'node:assert/strict';
import test from 'node:test';
import { getGlossaryEntry, PIPELINE_GLOSSARY } from './glossary.ts';
import {
  getNextManualTheme,
  parseThemePreference,
  resolveTheme,
} from './theme.ts';

test('resolveTheme respects system preference when set to system', () => {
  assert.equal(resolveTheme('system', true), 'dark');
  assert.equal(resolveTheme('system', false), 'light');
  assert.equal(resolveTheme('dark', false), 'dark');
  assert.equal(resolveTheme('light', true), 'light');
});

test('parseThemePreference and getNextManualTheme handle defaults and flips safely', () => {
  assert.equal(parseThemePreference(null), 'system');
  assert.equal(parseThemePreference('invalid-value'), 'system');
  assert.equal(parseThemePreference('dark'), 'dark');
  assert.equal(parseThemePreference('light'), 'light');
  assert.equal(getNextManualTheme('light'), 'dark');
  assert.equal(getNextManualTheme('dark'), 'light');
});

test('pipeline glossary includes S-invariant and Flanking-KL divergence definitions', () => {
  const sInv = getGlossaryEntry('s-invariant');
  const flankKl = getGlossaryEntry('flanking-kl');
  assert.ok(sInv && sInv.summary.length > 40);
  assert.ok(flankKl && flankKl.summary.includes('Kullback–Leibler'));
  assert.ok(Object.keys(PIPELINE_GLOSSARY).length >= 8);
});
