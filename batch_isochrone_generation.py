#!/usr/bin/env python3
"""
Batch Isochrone Generation Script

This script generates isochrones for both stroke units and CT hospitals
using the enhanced GeoStroke isochrone generation system.

Usage:
    python batch_isochrone_generation.py [--force-recalc] [--test-only]
"""

import argparse
import sys
from pathlib import Path

import geostroke as gs
from geostroke import iso_manager, data

def main():
    parser = argparse.ArgumentParser(description='Generate isochrones for GeoStroke analysis')
    parser.add_argument('--force-recalc', action='store_true', 
                       help='Force recalculation even if cached polygons exist')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test connectivity, do not generate isochrones')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts for failed requests')
    
    args = parser.parse_args()
    
    print("🗺️ GeoStroke Batch Isochrone Generation")
    print("=" * 50)
    
    # Test ORS connectivity
    print("1. Testing OpenRouteService connectivity...")
    if not iso_manager.test_ors_connectivity():
        print("❌ OpenRouteService not available!")
        print("   Please start ORS server or check configuration")
        return 1
    print("✅ OpenRouteService is ready")
    
    if args.test_only:
        print("\n🧪 Test mode only - exiting")
        return 0
    
    # Load datasets
    print("\n2. Loading datasets...")
    try:
        df_stroke = data.load_stroke_units()
        df_ct = data.load_hospitals_ct()
        print(f"✅ Loaded {len(df_stroke)} stroke units")
        print(f"✅ Loaded {len(df_ct)} CT hospitals")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Prepare datasets for batch processing
    datasets = {
        "Stroke Units": (df_stroke, ""),
        "CT Hospitals": (df_ct, "_all_CTs")
    }
    
    print(f"\n3. Generating isochrones...")
    print(f"   Force recalc: {args.force_recalc}")
    print(f"   Max retries: {args.max_retries}")
    
    # Batch generate isochrones
    results = iso_manager.batch_generate_isochrones(
        datasets, 
        force_recalc=args.force_recalc,
        max_retries=args.max_retries
    )
    
    # Summary
    print("\n🎯 Generation Summary:")
    for name, polygons in results.items():
        if polygons:
            total_polys = sum(len(plist) for plist in polygons.values())
            print(f"✅ {name}: {total_polys} polygons generated")
        else:
            print(f"❌ {name}: Generation failed")
    
    success = all(polygons for polygons in results.values())
    if success:
        print("\n🎉 All isochrone generation completed successfully!")
        return 0
    else:
        print("\n⚠️ Some isochrone generation failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
