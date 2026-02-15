#![feature(offset_of_enum)]

//! Entity-component system stored in a region.
//!
//! Demonstrates all-primitive structs in `NearList`, `map_list` for
//! physics/damage ticks, `filter_list` to cull dead entities,
//! `push_back` with `ListTail` for O(1) repeated appends, `set` for
//! scalar mutation, and `trim` for compaction.

use nearest::{Flat, NearList, Region};

#[derive(Flat, Copy, Clone, Debug, PartialEq)]
struct Entity {
  id: u32,
  x: i32,
  y: i32,
  vx: i32,
  vy: i32,
  hp: i32,
}

#[derive(Flat, Debug)]
struct World {
  tick: u32,
  entities: NearList<Entity>,
}

fn print_world(world: &World) {
  println!("tick {} ({} entities):", world.tick, world.entities.len());
  for e in &world.entities {
    println!("  [{}] pos=({},{}) vel=({},{}) hp={}", e.id, e.x, e.y, e.vx, e.vy, e.hp);
  }
}

fn main() {
  let mut region = Region::new(World::make(
    0,
    [
      Entity { id: 0, x: 0, y: 0, vx: 1, vy: 0, hp: 100 },
      Entity { id: 1, x: 10, y: 5, vx: -1, vy: 1, hp: 50 },
      Entity { id: 2, x: 3, y: 7, vx: 0, vy: -2, hp: 10 },
      Entity { id: 3, x: -5, y: 0, vx: 2, vy: 2, hp: 75 },
      Entity { id: 4, x: 20, y: 20, vx: -1, vy: -1, hp: 30 },
    ],
  ));

  println!("=== initial ===");
  print_world(&region);

  // Simulate one tick: apply velocity, deal damage, cull dead, spawn new.
  region.session(|s| {
    let entities = s.nav(s.root(), |w| &w.entities);

    // Physics: apply velocity to position.
    s.map_list(entities, |mut e| {
      e.x += e.vx;
      e.y += e.vy;
      e
    });

    // Damage: every entity takes 20 damage.
    s.map_list(entities, |mut e| {
      e.hp -= 20;
      e
    });

    // Cull dead entities (hp <= 0).
    s.filter_list(entities, |e| e.hp > 0);

    // Spawn two new entities using push_back with ListTail caching.
    let tail = s.push_back(entities, None, Entity { id: 10, x: 0, y: 0, vx: 3, vy: 1, hp: 100 });
    let _tail =
      s.push_back(entities, Some(tail), Entity { id: 11, x: -10, y: -10, vx: 1, vy: 1, hp: 80 });

    // Advance tick counter.
    let tick = s.nav(s.root(), |w| &w.tick);
    s.set(tick, 1);
  });

  println!("\n=== after tick 1 ===");
  print_world(&region);

  // Entities 2 and 4 should be dead (hp was 10-20=-10 and 30-20=10, wait
  // let me recalculate: entity 2 had hp=10, -20 = -10 dead; entity 4 had
  // hp=30, -20 = 10 alive).
  assert_eq!(region.tick, 1);
  // Alive: 0 (80hp), 1 (30hp), 3 (55hp), 4 (10hp), plus 2 new = 6 total.
  // Entity 2 (hp=10-20=-10) is dead.
  let alive: Vec<u32> = region.entities.iter().map(|e| e.id).collect();
  assert!(!alive.contains(&2), "entity 2 should be dead");
  assert!(alive.contains(&10), "entity 10 should be spawned");
  assert!(alive.contains(&11), "entity 11 should be spawned");

  // Trim compacts dead bytes.
  let before = region.byte_len();
  region.trim();
  println!("\ntrim: {} -> {} bytes", before, region.byte_len());

  // Data still intact after trim.
  assert_eq!(region.tick, 1);
  print_world(&region);
}
