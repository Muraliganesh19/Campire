import 'dart:math';
import 'package:flutter/material.dart';

class CameraGridOverlay extends StatelessWidget {
  const CameraGridOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _GridPainter(),
      child: Container(),
    );
  }
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.3)
      ..strokeWidth = 0.5;

    // Rule of thirds lines
    final double thirdWidth = size.width / 3;
    final double thirdHeight = size.height / 3;

    canvas.drawLine(Offset(thirdWidth, 0), Offset(thirdWidth, size.height), paint);
    canvas.drawLine(Offset(thirdWidth * 2, 0), Offset(thirdWidth * 2, size.height), paint);

    canvas.drawLine(Offset(0, thirdHeight), Offset(size.width, thirdHeight), paint);
    canvas.drawLine(Offset(0, thirdHeight * 2), Offset(size.width, thirdHeight * 2), paint);

    // Center reticle
    final center = Offset(size.width / 2, size.height / 2);
    final reticlePaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;
    
    // Segmented Circle (4 arcs)
    const double radius = 24.0;
    final rect = Rect.fromCircle(center: center, radius: radius);
    
    // Draw 4 arcs, leaving gaps
    const double sweepAngle = pi / 3; // 60 degrees arc
    const double gapAngle = pi / 6; // 30 degrees gap
    
    for (int i = 0; i < 4; i++) {
       double startAngle = (i * pi / 2) + (gapAngle / 2);
       canvas.drawArc(rect, startAngle, sweepAngle, false, reticlePaint);
    }

    // Fine crosshair (+)
    const double tick = 8.0;
    canvas.drawLine(Offset(center.dx - tick, center.dy), Offset(center.dx + tick, center.dy), reticlePaint);
    canvas.drawLine(Offset(center.dx, center.dy - tick), Offset(center.dx, center.dy + tick), reticlePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
