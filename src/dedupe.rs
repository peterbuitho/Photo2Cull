//! Phase 2 of the ranking system: detecting near-duplicate/burst-sequence
//! photos via perceptual hashing, and clustering them into groups.

use std::collections::HashMap;
use std::path::PathBuf;

use image::RgbImage;

/// 64-bit difference hash (dHash): resize to 9x8, grayscale, compare each
/// pixel to its right neighbor. Near-identical images -- duplicates,
/// consecutive burst frames -- produce hashes with a small Hamming
/// distance; visually different images diverge quickly. Cheap enough to
/// compute for every photo during the scan (negligible next to decoding).
pub fn dhash(rgb: &RgbImage) -> u64 {
    let small = image::imageops::resize(rgb, 9, 8, image::imageops::FilterType::Triangle);
    let gray = image::DynamicImage::ImageRgb8(small).to_luma8();

    let mut hash: u64 = 0;
    let mut bit = 0;
    for y in 0..8 {
        for x in 0..8 {
            let left = gray.get_pixel(x, y)[0];
            let right = gray.get_pixel(x + 1, y)[0];
            if left > right {
                hash |= 1 << bit;
            }
            bit += 1;
        }
    }
    hash
}

pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Clusters `photos` (path, dHash) into groups whose members are all
/// pairwise reachable within `max_distance` Hamming distance of each other
/// (via union-find, not just distance-to-the-first-member -- so a chain of
/// gradually-drifting near-duplicates still ends up in one group). Only
/// clusters with 2+ members are returned, largest first; unique photos
/// aren't "groups" and stay in the main grid.
///
/// O(n^2) hash comparisons, but each is a single XOR + popcount, so this
/// stays well under a second even for several thousand photos -- run
/// synchronously from a button click rather than backgrounded.
pub fn group_duplicates(photos: &[(PathBuf, u64)], max_distance: u32) -> Vec<Vec<PathBuf>> {
    let n = photos.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if hamming_distance(photos[i].1, photos[j].1) <= max_distance {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut groups: HashMap<usize, Vec<PathBuf>> = HashMap::new();
    for (i, photo) in photos.iter().enumerate() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(photo.0.clone());
    }

    let mut result: Vec<Vec<PathBuf>> = groups.into_values().filter(|g| g.len() > 1).collect();
    result.sort_by_key(|a| std::cmp::Reverse(a.len()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance_counts_differing_bits() {
        assert_eq!(hamming_distance(0b0000, 0b0000), 0);
        assert_eq!(hamming_distance(0b0000, 0b1111), 4);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }

    #[test]
    fn groups_close_hashes_and_excludes_singletons() {
        let photos = vec![
            (PathBuf::from("a"), 0b0000_0000u64),
            (PathBuf::from("b"), 0b0000_0001u64), // distance 1 from a
            (PathBuf::from("c"), 0b1111_1111u64), // distance 8 from a -- excluded
        ];
        let groups = group_duplicates(&photos, 2);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn chains_transitively_through_union_find() {
        // a-b close, b-c close, but a-c alone would exceed max_distance --
        // should still all land in one group via the b bridge.
        let photos = vec![
            (PathBuf::from("a"), 0b0000_0000u64),
            (PathBuf::from("b"), 0b0000_0011u64), // distance 2 from a
            (PathBuf::from("c"), 0b0000_1111u64), // distance 2 from b, distance 4 from a
        ];
        let groups = group_duplicates(&photos, 2);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }
}
