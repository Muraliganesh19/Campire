import 'package:flutter_test/flutter_test.dart';
import 'package:campire/network/clock_sync_service.dart';

void main() {
  // ═══════════════════════════════════════════════════════════════════════
  // ClockSyncSample — individual offset & RTT calculations
  // ═══════════════════════════════════════════════════════════════════════

  group('ClockSyncSample.offsetMicros', () {
    test('zero offset with symmetric delay', () {
      // Clocks are perfectly synchronised.
      // 10 µs flight time each way, no processing time on remote.
      //
      //   Controller    Remote
      //   T1 = 1000  ─▶  T2 = 1010   (flight = 10)
      //                   T3 = 1010  ─▶  T4 = 1020   (flight = 10)
      //
      // offset = ((1010-1000) + (1010-1020)) / 2 = (10 + (-10)) / 2 = 0
      const s = ClockSyncSample(t1: 1000, t2: 1010, t3: 1010, t4: 1020);
      expect(s.offsetMicros, 0);
    });

    test('positive offset — remote clock ahead', () {
      // Remote clock is 50 µs ahead. Flight time 10 µs each way.
      //
      //   Controller    Remote (ahead by 50)
      //   T1 = 1000  ─▶  T2 = 1060   (controller's view: 1010, + 50 offset)
      //                   T3 = 1060  ─▶  T4 = 1020
      //
      // offset = ((1060-1000) + (1060-1020)) / 2 = (60 + 40) / 2 = 50
      const s = ClockSyncSample(t1: 1000, t2: 1060, t3: 1060, t4: 1020);
      expect(s.offsetMicros, 50);
    });

    test('negative offset — remote clock behind', () {
      // Remote clock is 50 µs behind. Flight time 10 µs each way.
      //
      //   Controller    Remote (behind by 50)
      //   T1 = 1000  ─▶  T2 = 960    (controller's view: 1010, - 50 offset)
      //                   T3 = 960   ─▶  T4 = 1020
      //
      // offset = ((960-1000) + (960-1020)) / 2 = (-40 + -60) / 2 = -50
      const s = ClockSyncSample(t1: 1000, t2: 960, t3: 960, t4: 1020);
      expect(s.offsetMicros, -50);
    });

    test('asymmetric delay still computes correct offset', () {
      // Remote clock is 100 µs ahead.
      // Outbound flight = 20 µs, return flight = 5 µs (asymmetric).
      //
      //   Controller    Remote (ahead by 100)
      //   T1 = 1000  ─▶  T2 = 1120   (real arrival 1020 + 100 offset)
      //                   T3 = 1130  ─▶  T4 = 1035   (real arrival 1030+5, controller clock)
      //
      // NTP assumes symmetric delay, so the calculated offset is an
      // approximation. Let's verify the formula gives a consistent result:
      // offset = ((1120-1000) + (1130-1035)) / 2 = (120 + 95) / 2 = 107
      //
      // True offset is 100 — the error (7 µs) is expected with asymmetric
      // delay and is well within the 350 ms sync window.
      const s = ClockSyncSample(t1: 1000, t2: 1120, t3: 1130, t4: 1035);
      expect(s.offsetMicros, 107); // (120 + 95) ~/ 2
    });

    test('remote processing time does not affect offset', () {
      // Clocks perfectly synced. Flight = 10 each way.
      // Remote spends 500 µs processing between T2 and T3.
      //
      //   T1 = 1000  ─▶  T2 = 1010
      //                   T3 = 1510  ─▶  T4 = 1520
      //
      // offset = ((1010-1000) + (1510-1520)) / 2 = (10 + (-10)) / 2 = 0
      const s = ClockSyncSample(t1: 1000, t2: 1010, t3: 1510, t4: 1520);
      expect(s.offsetMicros, 0);
    });

    test('large offset values (millisecond range)', () {
      // Remote is 150 ms (150000 µs) behind. Flight = 50 ms each way.
      // (Matches the scaffold's original test case, now in µs.)
      //
      //   T1 = 1000000  ─▶  T2 =  900000   (1050000 - 150000)
      //                      T3 = 1000000  ─▶  T4 = 1200000
      //
      // offset = ((900000-1000000) + (1000000-1200000)) / 2
      //        = (-100000 + -200000) / 2 = -150000
      const s = ClockSyncSample(
        t1: 1000000,
        t2: 900000,
        t3: 1000000,
        t4: 1200000,
      );
      expect(s.offsetMicros, -150000);
    });
  });

  group('ClockSyncSample.roundTripMicros', () {
    test('symmetric delay', () {
      // Flight 10 each way, no processing.
      // RTT = (T4-T1) - (T3-T2) = (1020-1000) - (1010-1010) = 20
      const s = ClockSyncSample(t1: 1000, t2: 1010, t3: 1010, t4: 1020);
      expect(s.roundTripMicros, 20);
    });

    test('processing time excluded from RTT', () {
      // Flight 10 each way, 500 processing.
      // RTT = (1520-1000) - (1510-1010) = 520 - 500 = 20
      const s = ClockSyncSample(t1: 1000, t2: 1010, t3: 1510, t4: 1520);
      expect(s.roundTripMicros, 20);
    });

    test('asymmetric delay RTT', () {
      // Outbound 20, return 5, processing 10.
      // T1=1000, T2=1120, T3=1130, T4=1035  (from offset test above)
      // RTT = (1035-1000) - (1130-1120) = 35 - 10 = 25
      const s = ClockSyncSample(t1: 1000, t2: 1120, t3: 1130, t4: 1035);
      expect(s.roundTripMicros, 25);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // computeAverageOffset — outlier filtering + averaging
  // ═══════════════════════════════════════════════════════════════════════

  group('computeAverageOffset', () {
    test('averages offsets after discarding highest-RTT samples', () {
      // 5 samples, discard 2 highest RTT.
      // We'll construct samples with known offsets and RTTs.
      final samples = [
        // offset=100, RTT=10  ← keep
        const ClockSyncSample(t1: 0, t2: 105, t3: 105, t4: 10),
        // offset=100, RTT=20  ← keep
        const ClockSyncSample(t1: 0, t2: 110, t3: 110, t4: 20),
        // offset=100, RTT=30  ← keep
        const ClockSyncSample(t1: 0, t2: 115, t3: 115, t4: 30),
        // offset=200, RTT=100 ← discard (high RTT)
        const ClockSyncSample(t1: 0, t2: 250, t3: 250, t4: 100),
        // offset=300, RTT=200 ← discard (highest RTT)
        const ClockSyncSample(t1: 0, t2: 400, t3: 400, t4: 200),
      ];

      // Verify our RTT expectations.
      expect(samples[0].roundTripMicros, 10);
      expect(samples[1].roundTripMicros, 20);
      expect(samples[2].roundTripMicros, 30);
      expect(samples[3].roundTripMicros, 100);
      expect(samples[4].roundTripMicros, 200);

      // After discarding 2 highest-RTT: keep samples 0, 1, 2.
      // Their offsets: 100, 100, 100 → average = 100.
      final avg = computeAverageOffset(samples, discardCount: 2);
      expect(avg, 100);
    });

    test('handles mixed positive and negative offsets', () {
      final samples = [
        // offset=50, RTT=10
        const ClockSyncSample(t1: 0, t2: 55, t3: 55, t4: 10),
        // offset=-30, RTT=20
        const ClockSyncSample(t1: 0, t2: -20, t3: -20, t4: 20),
        // offset=80, RTT=15
        const ClockSyncSample(t1: 0, t2: 87, t3: 88, t4: 15),
        // offset=999, RTT=500 ← discard
        const ClockSyncSample(t1: 0, t2: 1249, t3: 1249, t4: 500),
      ];

      // Keep 3, discard 1. Offsets of kept: 50, -30, 80 → avg = 33
      // Let me re-check: sample[2] offset = ((87-0) + (88-15)) / 2 = (87+73)/2 = 80
      expect(samples[2].offsetMicros, 80);
      
      final avg = computeAverageOffset(samples, discardCount: 1);
      // (50 + (-30) + 80) / 3 = 100 / 3 = 33 (integer division)
      expect(avg, 33);
    });

    test('single discard from three samples', () {
      final samples = [
        const ClockSyncSample(t1: 0, t2: 55, t3: 55, t4: 10), // offset=50, RTT=10
        const ClockSyncSample(t1: 0, t2: 55, t3: 55, t4: 10), // offset=50, RTT=10
        const ClockSyncSample(t1: 0, t2: 300, t3: 300, t4: 100), // offset=250, RTT=100 ← discard
      ];

      final avg = computeAverageOffset(samples, discardCount: 1);
      expect(avg, 50);
    });

    test('throws when not enough samples after discarding', () {
      final samples = [
        const ClockSyncSample(t1: 0, t2: 55, t3: 55, t4: 10),
        const ClockSyncSample(t1: 0, t2: 55, t3: 55, t4: 10),
      ];

      expect(
        () => computeAverageOffset(samples, discardCount: 2),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('throws on empty sample list', () {
      expect(
        () => computeAverageOffset([], discardCount: 1),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('discarding zero outliers averages all samples', () {
      final samples = [
        const ClockSyncSample(t1: 0, t2: 55, t3: 55, t4: 10), // offset=50
        const ClockSyncSample(t1: 0, t2: 155, t3: 155, t4: 10), // offset=150
      ];

      final avg = computeAverageOffset(samples, discardCount: 0);
      expect(avg, 100); // (50 + 150) / 2
    });

    test('realistic scenario: 8 samples, discard 3', () {
      // Simulate a real-world scenario with a true offset of ~200 µs.
      // Low-RTT samples should cluster near the true offset; high-RTT
      // samples will be noisier.
      final samples = [
        const ClockSyncSample(t1: 0, t2: 210, t3: 210, t4: 20), // off=200, RTT=20
        const ClockSyncSample(t1: 0, t2: 215, t3: 215, t4: 30), // off=200, RTT=30
        const ClockSyncSample(t1: 0, t2: 208, t3: 208, t4: 16), // off=200, RTT=16
        const ClockSyncSample(t1: 0, t2: 212, t3: 212, t4: 24), // off=200, RTT=24
        const ClockSyncSample(t1: 0, t2: 205, t3: 205, t4: 10), // off=200, RTT=10
        // ── noisy samples below ──
        const ClockSyncSample(t1: 0, t2: 350, t3: 350, t4: 200), // off=250, RTT=100 ← discard
        const ClockSyncSample(t1: 0, t2: 550, t3: 550, t4: 500), // off=300, RTT=400 ← discard
        const ClockSyncSample(t1: 0, t2: 900, t3: 900, t4: 800), // off=500, RTT=700 ← discard
      ];

      final avg = computeAverageOffset(samples, discardCount: 3);
      // Kept samples all have offset 200 → average = 200
      expect(avg, 200);
    });

    test('order of samples does not affect result', () {
      // Same data as previous test but shuffled.
      final samples = [
        const ClockSyncSample(t1: 0, t2: 900, t3: 900, t4: 800), // RTT=700
        const ClockSyncSample(t1: 0, t2: 205, t3: 205, t4: 10), // RTT=10
        const ClockSyncSample(t1: 0, t2: 350, t3: 350, t4: 200), // RTT=100
        const ClockSyncSample(t1: 0, t2: 210, t3: 210, t4: 20), // RTT=20
        const ClockSyncSample(t1: 0, t2: 550, t3: 550, t4: 500), // RTT=400
        const ClockSyncSample(t1: 0, t2: 215, t3: 215, t4: 30), // RTT=30
        const ClockSyncSample(t1: 0, t2: 208, t3: 208, t4: 16), // RTT=16
        const ClockSyncSample(t1: 0, t2: 212, t3: 212, t4: 24), // RTT=24
      ];

      final avg = computeAverageOffset(samples, discardCount: 3);
      expect(avg, 200);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Edge cases
  // ═══════════════════════════════════════════════════════════════════════

  group('Edge cases', () {
    test('identical timestamps yield zero offset and zero RTT', () {
      const s = ClockSyncSample(t1: 5000, t2: 5000, t3: 5000, t4: 5000);
      expect(s.offsetMicros, 0);
      expect(s.roundTripMicros, 0);
    });

    test('very large timestamp values (realistic epoch microseconds)', () {
      // Approx 2025-07-01 in microseconds since epoch.
      const base = 1751328000000000;
      const offset = 1234; // 1.234 ms offset

      // Flight = 500 µs each way, processing = 100 µs.
      const s = ClockSyncSample(
        t1: base,
        t2: base + 500 + offset,       // arrived at remote
        t3: base + 500 + 100 + offset,  // remote sends response
        t4: base + 500 + 100 + 500,     // arrived back at controller
      );

      expect(s.offsetMicros, offset);
      expect(s.roundTripMicros, 1000); // 500 + 500
    });

    test('integer division truncation is consistent', () {
      // Odd sum that doesn't divide evenly.
      // offset = ((11 - 0) + (11 - 20)) / 2 = (11 + (-9)) / 2 = 2 / 2 = 1
      const s = ClockSyncSample(t1: 0, t2: 11, t3: 11, t4: 20);
      expect(s.offsetMicros, 1);

      // Another: offset = ((10 - 0) + (10 - 21)) / 2 = (10 + (-11)) / 2 = -1 / 2 = 0
      const s2 = ClockSyncSample(t1: 0, t2: 10, t3: 10, t4: 21);
      expect(s2.offsetMicros, 0); // truncated toward zero
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Backward compatibility: verify the old scaffold's test cases still
  // hold (converted from ms to µs).
  // ═══════════════════════════════════════════════════════════════════════

  group('Backward compatibility with scaffold tests', () {
    test('positive offset (original scaffold case)', () {
      // Original: T0=1000ms, T1=1100ms, T2=1200ms, T3=1200ms → offset=+50ms
      // In µs: T1=1000000, T2=1100000, T3=1200000, T4=1200000
      const s = ClockSyncSample(
        t1: 1000000,
        t2: 1100000,
        t3: 1200000,
        t4: 1200000,
      );
      expect(s.offsetMicros, 50000); // 50 ms in µs
    });

    test('negative offset (original scaffold case)', () {
      // Original: T0=1000ms, T1=900ms, T2=1000ms, T3=1200ms → offset=-150ms
      const s = ClockSyncSample(
        t1: 1000000,
        t2: 900000,
        t3: 1000000,
        t4: 1200000,
      );
      expect(s.offsetMicros, -150000); // -150 ms in µs
    });

    test('round trip time (original scaffold case)', () {
      // Original: T0=1000ms, T3=1350ms → RTT=350ms
      // We need T2 and T3 for the new formula: RTT = (T4-T1) - (T3-T2)
      // Use T2=T3 (no processing): RTT = (1350000-1000000) - 0 = 350000
      const s = ClockSyncSample(
        t1: 1000000,
        t2: 1175000, // T2 value doesn't matter for RTT
        t3: 1175000, // T3 = T2 (no processing)
        t4: 1350000,
      );
      expect(s.roundTripMicros, 350000); // 350 ms in µs
    });
  });
}
