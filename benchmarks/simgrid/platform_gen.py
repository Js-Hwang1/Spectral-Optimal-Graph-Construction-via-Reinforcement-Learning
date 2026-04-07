#!/usr/bin/env python3
"""
platform_gen.py — Generate SimGrid XML platform files for datacenter simulations.

Configurations:
  3tier   - 3-tier bandwidth hierarchy (rack/pod)
  fattree - k-ary fat-tree
  homo    - Homogeneous (all links identical)

Usage:
  python3 platform_gen.py 3tier   64 > platform_3tier_64.xml
  python3 platform_gen.py fattree 64 --k 4 > platform_fattree_64.xml
  python3 platform_gen.py homo    64 > platform_homo_64.xml
"""

import sys
import argparse
import math
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def prettify(elem):
    raw = tostring(elem, encoding="unicode")
    parsed = minidom.parseString(raw)
    lines = parsed.toprettyxml(indent="  ").split("\n")
    # Remove the XML declaration minidom adds, we'll write our own
    return "\n".join(lines[1:])


def gen_3tier(n, rack_size=8, racks_per_pod=4):
    """
    3-tier datacenter:
      - Intra-rack:  100 Gbps, 1us latency
      - Inter-rack:  10 Gbps,  10us latency
      - Cross-pod:   1 Gbps,   100us latency
    """
    num_racks = math.ceil(n / rack_size)
    num_pods = math.ceil(num_racks / racks_per_pod)

    platform = Element("platform", version="4.1")
    zone = SubElement(platform, "zone", id="world", routing="Full")

    # Create hosts
    for i in range(n):
        SubElement(zone, "host", id=f"worker_{i}", speed="1Gf")

    # Create links
    # Backbone (cross-pod)
    SubElement(zone, "link", id="backbone",
               bandwidth="1Gbps", latency="100us")

    for pod_id in range(num_pods):
        # Pod aggregation link
        SubElement(zone, "link", id=f"pod_{pod_id}_agg",
                   bandwidth="10Gbps", latency="10us")

        for local_rack in range(racks_per_pod):
            rack_id = pod_id * racks_per_pod + local_rack
            if rack_id >= num_racks:
                break
            # Rack link
            SubElement(zone, "link", id=f"rack_{rack_id}",
                       bandwidth="100Gbps", latency="1us")

    # Create routes between all pairs
    for i in range(n):
        rack_i = i // rack_size
        pod_i = rack_i // racks_per_pod
        for j in range(i + 1, n):
            rack_j = j // rack_size
            pod_j = rack_j // racks_per_pod

            route = SubElement(zone, "route",
                               src=f"worker_{i}", dst=f"worker_{j}")

            if rack_i == rack_j:
                # Same rack
                SubElement(route, "link_ctn", id=f"rack_{rack_i}")
            elif pod_i == pod_j:
                # Same pod, different rack
                SubElement(route, "link_ctn", id=f"rack_{rack_i}")
                SubElement(route, "link_ctn", id=f"pod_{pod_i}_agg")
                SubElement(route, "link_ctn", id=f"rack_{rack_j}")
            else:
                # Different pod
                SubElement(route, "link_ctn", id=f"rack_{rack_i}")
                SubElement(route, "link_ctn", id=f"pod_{pod_i}_agg")
                SubElement(route, "link_ctn", id="backbone")
                SubElement(route, "link_ctn", id=f"pod_{pod_j}_agg")
                SubElement(route, "link_ctn", id=f"rack_{rack_j}")

    return platform


def gen_fattree(n, k=4):
    """
    k-ary fat-tree topology.
    For simplicity, we model it as a 2-level hierarchy:
      - k pods, each with k/2 hosts
      - Edge links: 10 Gbps, 1us
      - Aggregation links: 10 Gbps, 5us
      - Core links: 40 Gbps, 10us
    Total hosts = k * k (we map n workers onto these)
    """
    hosts_per_pod = max(1, math.ceil(n / k))

    platform = Element("platform", version="4.1")
    zone = SubElement(platform, "zone", id="world", routing="Full")

    # Create hosts
    for i in range(n):
        SubElement(zone, "host", id=f"worker_{i}", speed="1Gf")

    # Core link
    SubElement(zone, "link", id="core",
               bandwidth="40Gbps", latency="10us")

    for pod_id in range(k):
        # Aggregation link per pod
        SubElement(zone, "link", id=f"agg_{pod_id}",
                   bandwidth="10Gbps", latency="5us")
        # Edge link per pod
        SubElement(zone, "link", id=f"edge_{pod_id}",
                   bandwidth="10Gbps", latency="1us")

    # Routes
    for i in range(n):
        pod_i = i // hosts_per_pod
        if pod_i >= k:
            pod_i = k - 1
        for j in range(i + 1, n):
            pod_j = j // hosts_per_pod
            if pod_j >= k:
                pod_j = k - 1

            route = SubElement(zone, "route",
                               src=f"worker_{i}", dst=f"worker_{j}")

            if pod_i == pod_j:
                # Intra-pod
                SubElement(route, "link_ctn", id=f"edge_{pod_i}")
            else:
                # Inter-pod: edge -> agg -> core -> agg -> edge
                SubElement(route, "link_ctn", id=f"edge_{pod_i}")
                SubElement(route, "link_ctn", id=f"agg_{pod_i}")
                SubElement(route, "link_ctn", id="core")
                SubElement(route, "link_ctn", id=f"agg_{pod_j}")
                SubElement(route, "link_ctn", id=f"edge_{pod_j}")

    return platform


def gen_homo(n, bandwidth="10Gbps", latency="10us"):
    """
    Homogeneous platform: all links are identical.
    Each pair shares one link.
    """
    platform = Element("platform", version="4.1")
    zone = SubElement(platform, "zone", id="world", routing="Full")

    for i in range(n):
        SubElement(zone, "host", id=f"worker_{i}", speed="1Gf")

    # One shared link per pair
    SubElement(zone, "link", id="shared_link",
               bandwidth=bandwidth, latency=latency)

    for i in range(n):
        for j in range(i + 1, n):
            route = SubElement(zone, "route",
                               src=f"worker_{i}", dst=f"worker_{j}")
            SubElement(route, "link_ctn", id="shared_link")

    return platform


def main():
    parser = argparse.ArgumentParser(description="Generate SimGrid platform XML")
    parser.add_argument("config", choices=["3tier", "fattree", "homo"],
                        help="Datacenter configuration")
    parser.add_argument("n", type=int, help="Number of workers")
    parser.add_argument("--k", type=int, default=4,
                        help="Fat-tree arity (for fattree config)")
    parser.add_argument("--bandwidth", default="10Gbps",
                        help="Link bandwidth (for homo config)")
    parser.add_argument("--latency", default="10us",
                        help="Link latency (for homo config)")
    parser.add_argument("--rack-size", type=int, default=8,
                        help="Workers per rack (for 3tier)")
    parser.add_argument("--racks-per-pod", type=int, default=4,
                        help="Racks per pod (for 3tier)")

    args = parser.parse_args()

    if args.config == "3tier":
        platform = gen_3tier(args.n, args.rack_size, args.racks_per_pod)
    elif args.config == "fattree":
        platform = gen_fattree(args.n, args.k)
    elif args.config == "homo":
        platform = gen_homo(args.n, args.bandwidth, args.latency)

    # Output with doctype
    print('<?xml version="1.0"?>')
    print('<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">')
    print(prettify(platform))


if __name__ == "__main__":
    main()
