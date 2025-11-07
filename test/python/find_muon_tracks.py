#!/usr/bin/env python3
"""
Find muon tracks with n_track = 2 and n_points >= 6 in a ROOT file.
"""

import ROOT
import sys
from datetime import datetime
from pathlib import Path

# Load ROOT library
ROOT.gSystem.Load("libEDMUtil")

# Add the python path to import modules
sys.path.append("..")
from Fuc import Fuc


class MuonTrackFinder:
    def __init__(self, track_file_path):
        self.track_file_path = track_file_path
        self.fuc = Fuc()

    def find_muon_tracks(self):
        """Find muon tracks with n_track = 1 and n_points >= 3"""
        # Open the ROOT file
        track_file = ROOT.TFile.Open(self.track_file_path, "READ")
        if not track_file or track_file.IsZombie():
            print(f"Error: Could not open file {self.track_file_path}")
            return []

        # List available keys
        print("\nAvailable keys in ROOT file:")
        keys = track_file.GetListOfKeys()
        for i, key in enumerate(keys):
            if i < 10:  # Show first 10 keys
                print(f"  {key.GetName()}: {key.GetClassName()}")
            elif i == 10:
                print(f"  ... and {len(keys)-10} more")
                break

        # Get the TT tree
        track_tree = track_file.Get("TT")
        if not track_tree:
            print(f"Error: Could not find TT tree in {self.track_file_path}")
            track_file.Close()
            return []

        entries = track_tree.GetEntries()
        print(f"Searching for muon tracks in: {self.track_file_path}")
        print(f"Total entries in tree: {entries}")
        print("="*60)

        matching_tracks = []
        track1_points = []  # Store n_points for all n_track=1 tracks
        n_track_dist = {}  # Store distribution of n_track values

        # Loop through all entries
        for count in range(entries):
            track_tree.GetEntry(count)

            # Get track properties
            n_track = track_tree.NTracks
            n_points = track_tree.NTotPoints

            # Collect n_track distribution
            if n_track not in n_track_dist:
                n_track_dist[n_track] = 0
            n_track_dist[n_track] += 1

            # Collect statistics for n_track=1 tracks
            if n_track == 2:
                track1_points.append(n_points)

            # Check if criteria match
            if n_track == 2:  # First check if n_track = 1
                if n_points >= 6:  # Additional check for n_points
                    # Get track time
                    track_t = (
                        track_tree.start_TS.GetSec() * 1000000000
                        + track_tree.start_TS.GetNanoSec()
                    )

                    # Convert to readable datetime
                    track_datetime = datetime.fromtimestamp(track_t / 1e9)

                    # Get reference position and direction
                    ref_pos = [
                        track_tree.Coeff0[0],
                        track_tree.Coeff1[0],
                        track_tree.Coeff2[0],
                    ]
                    direction = [
                        track_tree.Coeff3[0],
                        track_tree.Coeff4[0],
                        track_tree.Coeff5[0],
                    ]

                    # Calculate track intersections with detector
                    intersection1, intersection2 = self.fuc.compute_track_points(
                        ref_pos, direction
                    )

                    # Skip if track doesn't go through CD
                    if intersection1 is None or intersection2 is None:
                        continue

                    # Store track information
                    track_info = {
                        'entry': count,
                        'time_ns': track_t,
                        'time_datetime': track_datetime,
                        'n_track': n_track,
                        'n_points': n_points,
                        'ref_pos': ref_pos,
                        'direction': direction,
                        'entry_point': intersection1,
                        'exit_point': intersection2,
                    }

                    matching_tracks.append(track_info)

                    # Print track information
                    print(f"\nFound matching track #{len(matching_tracks)}:")
                    print(f"  Entry: {count}")
                    print(f"  Time: {track_datetime}")
                    print(f"  Time (ns): {track_t}")
                    print(f"  NTracks: {n_track}")
                    print(f"  NPoints: {n_points}")
                    print(f"  Reference Position: ({ref_pos[0]:.2f}, {ref_pos[1]:.2f}, {ref_pos[2]:.2f})")
                    print(f"  Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
                    print(f"  Entry Point: ({intersection1[0]:.2f}, {intersection1[1]:.2f}, {intersection1[2]:.2f})")
                    print(f"  Exit Point: ({intersection2[0]:.2f}, {intersection2[1]:.2f}, {intersection2[2]:.2f})")

                    # Calculate track length
                    track_length = (
                        (intersection2[0] - intersection1[0])**2 +
                        (intersection2[1] - intersection1[1])**2 +
                        (intersection2[2] - intersection1[2])**2
                    )**0.5
                    print(f"  Track Length: {track_length:.2f} mm")

        # Close the file
        track_file.Close()

        # Print summary
        print("\n" + "="*60)
        print(f"SUMMARY:")
        print(f"Total tracks processed: {entries}")
        print(f"Tracks with n_track=1: {len(track1_points)}")
        print(f"Tracks matching criteria (n_track=1, n_points>=3): {len(matching_tracks)}")

        # Print n_track distribution
        print(f"\nNTrack distribution:")
        for n_track, count in sorted(n_track_dist.items()):
            print(f"  n_track={n_track}: {count} tracks ({count/entries*100:.1f}%)")

        if track1_points:
            print(f"\nStatistics for n_track=1 tracks:")
            print(f"  Min n_points: {min(track1_points)}")
            print(f"  Max n_points: {max(track1_points)}")
            print(f"  Avg n_points: {sum(track1_points)/len(track1_points):.2f}")
            print(f"  Tracks with n_points < 6: {sum(1 for p in track1_points if p < 6)}")
            print(f"  Tracks with n_points >= 6: {sum(1 for p in track1_points if p >= 6)}")

        if matching_tracks:
            print(f"\nFirst matching track: {matching_tracks[0]['time_datetime']}")
            print(f"Last matching track: {matching_tracks[-1]['time_datetime']}")

        return matching_tracks

    def save_results(self, tracks, output_file=None):
        """Save results to a text file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"muon_tracks_ntrack1_npoints3_{timestamp}.txt"

        with open(output_file, 'w') as f:
            f.write(f"Muon tracks with n_track=1 and n_points>=3\n")
            f.write(f"Source file: {self.track_file_path}\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write(f"Total matching tracks: {len(tracks)}\n")
            f.write("="*60 + "\n\n")

            for i, track in enumerate(tracks):
                f.write(f"Track #{i+1}:\n")
                f.write(f"  Entry: {track['entry']}\n")
                f.write(f"  Time: {track['time_datetime']}\n")
                f.write(f"  Time (ns): {track['time_ns']}\n")
                f.write(f"  NTracks: {track['n_track']}\n")
                f.write(f"  NPoints: {track['n_points']}\n")
                f.write(f"  Reference Position: ({track['ref_pos'][0]:.2f}, {track['ref_pos'][1]:.2f}, {track['ref_pos'][2]:.2f})\n")
                f.write(f"  Direction: ({track['direction'][0]:.2f}, {track['direction'][1]:.2f}, {track['direction'][2]:.2f})\n")
                f.write(f"  Entry Point: ({track['entry_point'][0]:.2f}, {track['entry_point'][1]:.2f}, {track['entry_point'][2]:.2f})\n")
                f.write(f"  Exit Point: ({track['exit_point'][0]:.2f}, {track['exit_point'][1]:.2f}, {track['exit_point'][2]:.2f})\n")

                # Calculate track length
                track_length = (
                    (track['exit_point'][0] - track['entry_point'][0])**2 +
                    (track['exit_point'][1] - track['entry_point'][1])**2 +
                    (track['exit_point'][2] - track['entry_point'][2])**2
                )**0.5
                f.write(f"  Track Length: {track_length:.2f} mm\n\n")

        print(f"\nResults saved to: {output_file}")


def main():
    # Path to the ROOT file
    track_file_path = "/data/juno/lin/JUNO/draw/muon_track_reco_transformer/test_data/20250917_v1/muonTT_select_run_9737_20250917_v1.root"

    # Create finder instance
    finder = MuonTrackFinder(track_file_path)

    # Find matching tracks
    matching_tracks = finder.find_muon_tracks()

    # Save results
    if matching_tracks:
        finder.save_results(matching_tracks)
    else:
        print("\nNo tracks found matching the criteria.")


if __name__ == "__main__":
    main()