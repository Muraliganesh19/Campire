import 'dart:io';

/// Where the host stores session files and finds the analysis script.
///
/// Set `CAMPIRE_BASE_DIR` or `CAMPIRE_VIDEOMAKER` in the environment to
/// override the defaults, for example on a Windows PC with no `E:` drive.
class HostPaths {
  HostPaths._();

  static String get _home => Platform.environment['HOME'] ?? '.';
  static String get _sep => Platform.pathSeparator;

  /// Parent of the `Input` and `Output` session folders.
  static final String baseDir =
      Platform.environment['CAMPIRE_BASE_DIR'] ??
      (Platform.isWindows ? r'E:\campire' : '$_home/campire');

  static String get inputDir => '$baseDir${_sep}Input';
  static String get outputDir => '$baseDir${_sep}Output';

  /// The script ANALYZE runs for file sessions.
  static final String videomakerScript =
      Platform.environment['CAMPIRE_VIDEOMAKER'] ??
      (Platform.isWindows
          ? r'C:\Users\MURALI GANESH\OneDrive\Desktop\vdor\videomaker.py'
          : '$_home/mg/Campire/simulation/videomaker.py');

  /// Python interpreter name for this platform.
  static String get pythonCommand => Platform.isWindows ? 'python' : 'python3';
}
