use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Item {
    pub id: String,
    pub name: String,
    pub short_name: String,
    pub width: u32,
    pub height: u32,
    pub background_color: String,
    pub grid_image_link: String,
}

pub struct Catalog {
    pub items: Vec<Item>,
    by_id: HashMap<String, usize>,
}

impl Catalog {
    pub fn load(refs_dir: &Path) -> Result<Self> {
        let path = refs_dir.join("catalog.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Self::new(serde_json::from_str(&text)?))
    }

    pub fn new(items: Vec<Item>) -> Self {
        let by_id = items.iter().enumerate().map(|(i, it)| (it.id.clone(), i)).collect();
        Self { items, by_id }
    }

    pub fn get(&self, id: &str) -> Option<&Item> {
        self.by_id.get(id).map(|&i| &self.items[i])
    }

    pub fn index(&self, id: &str) -> Option<usize> {
        self.by_id.get(id).copied()
    }

    pub fn short_name<'a>(&'a self, id: &'a str) -> &'a str {
        self.get(id).map(|i| i.short_name.as_str()).unwrap_or(id)
    }

    /// Indices of items whose size matches `slots` as-is (rotated = false) or transposed (rotated = true).
    pub fn candidates(&self, slots: (u32, u32)) -> Vec<(usize, bool)> {
        let (w, h) = slots;
        let mut out = Vec::new();
        for (i, it) in self.items.iter().enumerate() {
            if it.width == w && it.height == h {
                out.push((i, false));
            } else if w != h && it.width == h && it.height == w {
                out.push((i, true));
            }
        }
        out
    }
}
