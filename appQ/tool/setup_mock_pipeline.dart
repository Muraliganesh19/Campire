import 'dart:io';

void main() async {
  final tempPath = Directory.systemTemp.path + Platform.pathSeparator + 'campire_mock';
  final tempDir = Directory(tempPath);
  if (!await tempDir.exists()) {
    await tempDir.create(recursive: true);
  }

  final mockDecisionFile = File(tempPath + Platform.pathSeparator + 'mock_decision.txt');
  await mockDecisionFile.writeAsString('''ball_decision=OUT
bowler_decision=LEGAL
''');
  print('Created ${mockDecisionFile.path}');

  for (int i = 1; i <= 4; i++) {
    final mockVideoFile = File(tempPath + Platform.pathSeparator + 'mock_cam$i.avi');
    // Writing 1 dummy byte so file isn't perfectly empty
    await mockVideoFile.writeAsBytes([0x00]);
    print('Created dummy video at ${mockVideoFile.path}');
  }
  
  print('Mock pipeline setup complete.');
}
