import 'dart:io';
import 'dart:typed_data';

/// Network configuration constants for discovery and command channels.
///
/// Ports are in the 45000+ range to avoid conflicts with common services.
/// Both controller and remotes must agree on these values.
class NetworkConstants {
  NetworkConstants._(); // prevent instantiation

  // ── Ports ──────────────────────────────────────────────────────────────

  /// UDP port used for discovery broadcasts and responses.
  static const int udpDiscoveryPort = 45010;

  /// TCP port used for the WebSocket command channel (start_at, ack, etc.).
  /// Not used by discovery or clock sync — reserved for command_channel.
  static const int tcpCommandPort = 45011;

  /// UDP port used for the NTP-style clock sync handshake.
  /// Separate from discovery so clock sync packets don't contend with
  /// broadcast traffic on the same socket.
  static const int udpClockSyncPort = 45012;

  // ── Discovery timing ───────────────────────────────────────────────────

  /// How often the controller broadcasts a discovery packet.
  static const Duration broadcastInterval = Duration(seconds: 1);

  /// How long a controller waits for remote responses before considering
  /// discovery "timed out" (used by UI, not enforced in the service itself).
  static const Duration discoveryTimeout = Duration(seconds: 10);

  // ── Clock sync timing ─────────────────────────────────────────────────

  /// Number of NTP-style round-trip samples to collect per device.
  static const int clockSyncSampleCount = 8;

  /// Number of highest-RTT samples to discard as outliers.
  static const int clockSyncDiscardCount = 3;

  /// How long to wait for a single clock_sync_response before considering
  /// the sample lost and moving on to the next one.
  static const Duration clockSyncSampleTimeout = Duration(milliseconds: 500);

  /// Small delay between consecutive clock sync samples to avoid packet
  /// bursts that could increase jitter.
  static const Duration clockSyncSampleGap = Duration(milliseconds: 50);

  // ── Broadcast address ──────────────────────────────────────────────────

  /// Computes the subnet-directed broadcast address for the first suitable
  /// IPv4 network interface.
  ///
  /// Returns the subnet broadcast (e.g. `192.168.1.255` for a `/24`) rather
  /// than `255.255.255.255`, because many Android hotspots and consumer
  /// routers silently drop limited-broadcast packets.
  ///
  /// Falls back to `255.255.255.255` if no usable interface is found.
  static Future<InternetAddress> broadcastAddress() async {
    final interfaces = await NetworkInterface.list(
      type: InternetAddressType.IPv4,
      includeLoopback: false,
      includeLinkLocal: false,
    );

    for (final iface in interfaces) {
      for (final addr in iface.addresses) {
        // Skip loopback and link-local even if the OS included them.
        if (addr.isLoopback || addr.isLinkLocal) continue;

        // Compute the subnet broadcast from the address bytes.
        // Android's NetworkInterface doesn't expose the netmask, so we
        // assume /24 — correct for the vast majority of home WiFi and
        // hotspot networks. A wrong assumption here means the broadcast
        // reaches a wider or narrower subnet, which is harmless for
        // discovery (worst case: packet is ignored or doesn't arrive,
        // and the user can fall back to manual IP entry later).
        final bytes = addr.rawAddress;
        final broadcast = InternetAddress.fromRawAddress(
          Uint8List.fromList([bytes[0], bytes[1], bytes[2], 255]),
        );
        return broadcast;
      }
    }

    // Fallback: limited broadcast (may be dropped by some routers).
    return InternetAddress('255.255.255.255');
  }

  /// Returns the first non-loopback IPv4 address of this device, or
  /// `0.0.0.0` if none is found.
  ///
  /// Used by the remote when constructing its [DeviceInfo] in a discovery
  /// response — the controller should prefer the packet's source IP, but
  /// the self-reported IP is useful for logging and debugging.
  static Future<InternetAddress> localAddress() async {
    final interfaces = await NetworkInterface.list(
      type: InternetAddressType.IPv4,
      includeLoopback: false,
      includeLinkLocal: false,
    );

    for (final iface in interfaces) {
      for (final addr in iface.addresses) {
        if (addr.isLoopback || addr.isLinkLocal) continue;
        return addr;
      }
    }

    return InternetAddress('0.0.0.0');
  }
}
