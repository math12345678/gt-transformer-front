import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GENOME_LENGTH_BP,
  MOSAIC_TRACKS,
  STRESS_TEST_BREAKPOINTS,
  Z_THRESHOLD,
  buildPolylinePath,
  buildTransformerPath,
  formatBp,
  generateGenomeScanProfile,
  projectBpToX,
  projectZToY,
} from './genomeScan';

test('stress test dataset contains all 10 sorted breakpoints within 0..12,000 bp', () => {
  assert.equal(STRESS_TEST_BREAKPOINTS.length, 10);
  for (let i = 0; i < STRESS_TEST_BREAKPOINTS.length; i++) {
    const ev = STRESS_TEST_BREAKPOINTS[i];
    assert.ok(ev.bp > 0 && ev.bp < GENOME_LENGTH_BP);
    assert.ok(ev.zScore > Z_THRESHOLD, `Event ${ev.id} should exceed z = ${Z_THRESHOLD}`);
    if (i > 0) {
      assert.ok(ev.bp > STRESS_TEST_BREAKPOINTS[i - 1].bp);
    }
  }
});

test('generated scan profile peaks above z = 6.0 at breakpoints and stays below in null regions', () => {
  const profile = generateGenomeScanProfile(STRESS_TEST_BREAKPOINTS, GENOME_LENGTH_BP, 50);
  assert.equal(profile[0].bp, 0);
  assert.equal(profile[profile.length - 1].bp, GENOME_LENGTH_BP);

  // Near breakpoint 1150 bp
  const peakPoint = profile.find((p) => p.bp === 1150);
  assert.ok(peakPoint && peakPoint.zScore > 12, 'Expected sharp peak at 1150 bp');
  assert.ok(peakPoint && peakPoint.tfProb > 0.9, 'Expected high Transformer probability at 1150 bp');

  // Far from any breakpoint (e.g. 6000 bp)
  const nullPoint = profile.find((p) => p.bp === 6000);
  assert.ok(nullPoint && nullPoint.zScore < Z_THRESHOLD, 'Expected null tract below z = 6.0');
  assert.ok(nullPoint && nullPoint.tfProb < 0.2, 'Expected low Transformer probability in null tract');
});

test('mosaic tracks tile the full 0..12,000 bp genome continuously with no gaps or overlaps', () => {
  assert.equal(MOSAIC_TRACKS.length, 4);
  for (const track of MOSAIC_TRACKS) {
    assert.equal(track.segments[0].startBp, 0);
    assert.equal(track.segments[track.segments.length - 1].endBp, GENOME_LENGTH_BP);
    for (let i = 1; i < track.segments.length; i++) {
      assert.equal(
        track.segments[i].startBp,
        track.segments[i - 1].endBp,
        `Track ${track.label} has discontinuity between segment ${i - 1} and ${i}`
      );
    }
  }
});

test('SVG coordinate projections clamp cleanly and emit valid SVG paths', () => {
  assert.equal(projectBpToX(0, 40, 600), 40);
  assert.equal(projectBpToX(12000, 40, 600), 640);
  assert.equal(projectBpToX(-100, 40, 600), 40);
  assert.equal(projectBpToX(15000, 40, 600), 640);

  assert.equal(projectZToY(0, 20, 170, 17), 190);
  assert.equal(projectZToY(17, 20, 170, 17), 20);

  const profile = generateGenomeScanProfile(STRESS_TEST_BREAKPOINTS, GENOME_LENGTH_BP, 200);
  const klPath = buildPolylinePath(profile, 40, 20, 600, 170);
  const tfPath = buildTransformerPath(profile, 40, 20, 600, 170);
  assert.ok(klPath.startsWith('M40.0,'));
  assert.ok(tfPath.startsWith('M40.0,'));
  assert.equal(formatBp(1150), '1,150 bp');
});
