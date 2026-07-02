# Pirate Fishing & Combat Roblox Game — Architecture Reference

> **Purpose:** Paste this into new Claude chats as context when working on this game. It covers the full architecture, conventions, key modules, data flow, and known patterns.

---

## Game Overview

A pirate-themed Roblox fishing and combat game with 4 island zones (Pirate, Greek, Japanese, Viking), ship combat with 6 ship tiers, 8 gadget throwables, melee weapons, and a deep fishing economy with mutations, rarities, and progressive rod upgrades.

**Solo developer (Ivar, Norwegian)** using Roblox Studio with MCP tooling for direct script editing via Claude.

---

## Global Architecture & Conventions

### `_G` Global System
All modules, remotes, and services are accessed via `_G` globals set up by `LoadGlobals`:

```
_G.MScript = {}          -- All client ModuleScripts (auto-loaded from ReplicatedStorage.Root)
_G.REvent = {}           -- All RemoteEvents (by name)
_G.RFunc = {}            -- All RemoteFunctions (by name)
_G.BEvent = {}           -- All BindableEvents (by name)
_G.EnumModule            -- Enums (ItemType, Rarity, RarityColor, RarityOrder, DataType, Mutation)
_G.Animations = {}       -- Animation objects
_G.Folders               -- Workspace folders (Projectiles, etc.)
_G.DataStore2            -- DataStore2 library (server only, but referenced everywhere)
_G.ServerMScript = {}    -- Server-side module references (set by server LoadGlobals)
_G.LoadingComplete       -- Set true after all Initialize/PostInitialize done
```

Services are also globalized: `_G.RunService`, `_G.TweenService`, `_G.Players`, `_G.CollectionService`, `_G.HttpService`, `_G.UIS`, `_G.Debris`, `_G.ReplicatedStorage`, `_G.ContentProvider`, `_G.ProximityPromptService`.

### Module Loading (LoadGlobals.lua — ReplicatedFirst)
1. Recursively walks `ReplicatedStorage.Root`, requiring every `ModuleScript` into `_G.MScript[name]`
2. Stores every `RemoteEvent` → `_G.REvent[name]`, `RemoteFunction` → `_G.RFunc[name]`, `BindableEvent` → `_G.BEvent[name]`
3. Calls `module.Initialize()` on every module that has it
4. Calls `module.PostInitialize()` on every module that has it (runs AFTER all Initialize, so LocalInventory is available)
5. Sets `_G.LoadingComplete = true`
6. Starts tutorial if first-time player

**Key rule:** If a module needs `LocalInventory` or other data that loads async, use `PostInitialize()` not `Initialize()`.

### Server-Side Globals (ServerScriptService.Root.DataManagement.LoadGlobals)
Sets up `_G.ServerMScript` with server-side modules (QuestHandler, PlayerMetricsHandler, etc.). Server scripts wait for `_G.LoadingComplete`.

### ScriptEditorService Workaround
Studio's script editor tab caching can override MCP `.Source` writes. Use `ScriptEditorService:CloseScriptDocument` when needed.

---

## Data Architecture

### DataStore2 (Combined Blob)
All 8 data stores are combined into a single DataStore2 blob per player under key `"DATA"`:

| DataType Key | Store Name | Description |
|---|---|---|
| INVENTORY | Inventory | Items, currencies, hotbar, bestRodLevel, ships storage |
| SHIPSDATA | ShipsData | OwnedShips, BestShip, ActiveShip, SlotProgress, SlotActives |
| FISHDEX | FishDex | Per-fish discovery data, zone species/mutation counts |
| OFFLINE_REWARDS | OfflineRewards | Fish cage, generation timing |
| PURCHASES | Purchases | permanent (bagTier, offlineExtend, sellAnywhere, safeStoragePurchaseCount), consumables |
| QUEST_DATA | QuestData | activeQuest, analytics, lastRefreshTime |
| PLAYER_METRICS | PlayerMetrics | fishing stats, combat, economy, sessions, tutorial, locationPath |
| SAFE_STORAGE | SafeStorage | items, purchaseCount |

**Critical:** Because all stores are combined, saving ANY store triggers a blob write. If module A updates its cache but module B saves its store, B's blob write may overwrite A's stale data. Fix: always call `store:Set(data)` after updating in-memory caches (we added `saveMetrics()` calls to fix this).

### Data Flow
- **Client → Server:** `DataModule.SetData(type, data)` (sync) or `DataModule.SetDataAsync(type, data)` (fire-and-forget via `task.spawn`)
- **Server → Client:** `_G.REvent.InventorySync:FireClient(player, inv)` (and similar per-store sync events)
- **Server DataHandler:** `ServerScriptService.Root.DataManagement.DataHandler` — handles `Get`/`Set` InvokeServer calls, player leaving saves, server closing saves
- **Server PlayerJoin:** `ServerScriptService.Root.DataManagement.PlayerJoin` — combines DataStore2 keys, loads all stores on join, sets up `OnUpdate` listeners for sync

### Optimistic Purchasing Pattern
Shop purchases use optimistic client-side updates:
1. Deduct gold locally → update UI instantly
2. Save async to server
3. `VerifyPurchaseAsync` calls server for validation → rollback on failure

---

## Key Server Scripts

| Script | Location | Purpose |
|---|---|---|
| PlayerJoin | SSS.Root.DataManagement.PlayerJoin | DataStore2 Combine, load all stores, OnUpdate sync, character spawn |
| DataHandler | SSS.Root.DataManagement.DataHandler | Get/Set InvokeServer, PlayerRemoving save, BindToClose |
| LoadGlobals (server) | SSS.Root.DataManagement.LoadGlobals | Sets up `_G.ServerMScript`, server module initialization |
| DevCommands | SSS.DevCommands | `/resetdata`, `/giverod`, etc. (49k chars, comprehensive) |
| QuestHandler | SSS.Root.EventHandlers.QuestHandler | Quest assignment, progression, completion, claiming, forced first quest |
| PlayerMetricsHandler | SSS.Root.EventHandlers.PlayerMetricsHandler | Tracks fishing, combat, economy, sessions, tutorial metrics |
| PurchaseHandler | SSS.Root.EventHandlers.PurchaseHandler | Server-side purchase verification |
| OfflineRewardsHandler | SSS.Root.EventHandlers.OfflineRewardsHandler | Offline fish generation |
| SafeStorageServer | SSS.SafeStorageServer | Safe storage chest management |
| ProductRegistry | RS.Root.DataManagement.ProductRegistry | Developer product definitions, ProcessReceipt |

---

## Key Client Modules (all in ReplicatedStorage.Root, accessed via `_G.MScript.ModuleName`)

### Core Systems
| Module | Size | Purpose |
|---|---|---|
| FishingModule | 94k | Main fishing loop: cast, reel, catch, loot floatables, AFK fishing |
| InventoryModule | 49k | LocalInventory cache, slot pooling, AddItem/RemoveItem, sorting |
| ShopModule | 36k | Rod/weapon/gadget shop UI, optimistic buy/upgrade/sell, auto-equip |
| ShipUpgradeModule | 66k | Ship weapon upgrades, bird cam, slot management |
| ShipShopModule | 17k | Ship purchase/activation UI |
| TutorialModule | 54k | 21-step tutorial with world arrows, UI arrows, ship billboards |
| QuestModule | 12k | Quest panel UI, progress bars, completion popup, refresh |
| QuestConfig | 15k | Quest generation, reward calculation, rarity targets |
| UpgradeProgressModule | 51k | Rod progress bar, upgrade beam system, ship guidance, shop icons |

### Economy & Data
| Module | Size | Purpose |
|---|---|---|
| FishGenerationModule | 21k | Gaussian fish generation, rarity RNG, mutation system, two-roll mutation events |
| CurrencyModule | 3k | Gold display, HasEnoughGold, DeductGoldLocal, VerifyPurchaseAsync |
| DataModule | 2k | Client-side Get/Set data via InvokeServer |
| SellFishModule | 15k | Sell backpack, sell all, sell individual fish |
| ProductRegistry | 12k | Developer products (bag space, sell anywhere, offline extend, safe storage) |

### Combat
| Module | Size | Purpose |
|---|---|---|
| DamageModule | 7k | Centralized damage, IsInSafeZone, PvP zone checks |
| MeleeWeaponModule | 8k | Sword/axe/spear combat, combos, animation sequencing |
| GadgetModule | 9k | 8 throwable gadgets (smoke, bear trap, tar, pulse, frost, pufferfish, ink, fire) |
| PlayerHealthBar | 15k | Custom overhead healthbars, shield, damage numbers |
| PlayerState | 2k | Status effects (frozen, stunned) |

### Ships & Weapons
| Module | Size | Purpose |
|---|---|---|
| ShipManager | 16k | Ship spawning, force-based movement, collision |
| CannonModule | 29k | Cannon firing, velocity prediction, PREDICTIVE_AIMING |
| HarpoonModule | 49k | Harpoon seats, rope physics, player yoinking |
| BallistaModule | 19k | Ballista firing mechanics |
| MortarModule | 13k | Mortar arc firing |
| SwivelModule | 13k | Swivel gun mechanics |
| SpawnZoneGuard | 10k | Spawn zone protection, countdown billboards |

### UI & Utility
| Module | Size | Purpose |
|---|---|---|
| UIManager | 2k | Exclusive screen system (Open/Close/Toggle/Register) |
| HotbarModule | 25k | Hotbar slots, drag behavior, equip/unequip |
| FishDexModule | 15k | Fish discovery collection UI |
| HUDModule | 16k | Main HUD elements |
| UtilityModule | 23k | FormatCurrency, DeepCopy, SetAttributes, GetRarityFromItemPower, etc. |
| EnumModule | 7k | All game enums: ItemType, Rarity, RarityColor, DataType, Mutation |
| CoinFlyModule | 7k | Gold coin fly animation on purchases/rewards |

### Environment
| Module | Size | Purpose |
|---|---|---|
| WaveModule | 32k | Ocean wave generation (4356 nodes), calm zones |
| MutationEventModule | 45k | Timed mutation events per zone, obelisk countdowns, two-roll system |
| ZoneConfig | 5k | Zone definitions (Pirate/Greek/Japanese/Viking), rod level ranges, rarity RNG |
| ZoneUIModule | 3k | Zone detection for UI triggers (shops, sell fish, portals) |
| FloatableModule | 20k | Floating loot barrels at sea |

---

## Zone System

4 zones with progressive rod level requirements:
- **Pirate:** Rod levels 1–6 (Rod IDs 1–2)
- **Greek:** Rod levels 7–12 (Rod IDs 3–4)
- **Japanese:** Rod levels 13–21 (Rod IDs 5–7)
- **Viking:** Rod levels 22–30 (Rod IDs 8–10)

Each rod has 3 upgrade levels. `bestRodLevel` = `(rodId - 1) * 3 + upgradeLevel`. Max is 30.

Fish rarities: Poor → Common → Uncommon → Rare → Epic → Legendary → Mythic.
Poor fish only spawn in safe zones (rerolled to Common outside).

---

## Hint/Beam Guidance System

Centralized `HintBanner` system (`StarterGui.HintBanner.Canvas.HintLabel`):
- `_G.ShowHintBanner(key, text, priority, duration)` — lower number = higher priority
- `_G.HideHintBanner(key)` — removes hint, next priority shows
- Priority 1: Upgrade rod/ship hints (from UpgradeProgressModule beam system)
- Priority 2: Zone level warning ("below level for this island")
- Priority 3: Safe zone fishing warning (15s duration)
- All hints blocked during tutorial (`TutorialComplete` attribute check)

**Upgrade Beam System** (in UpgradeProgressModule):
- Gold beam guides player to fish seller (sell mode) or rod/ship shop (buy mode)
- Ship takes priority over rod when player can afford next unowned ship
- `MainRodShop`, `MainFishSeller`, `MainShipSeller` CollectionService tags (4 each, one per island)
- Tutorial-style shop icons (`TutorialIcon_FirstItem` / `TutorialIcon_FirstShip`) on target frames

---

## Quest System

- **QuestHandler** (server): assignment, progression, completion, claiming
- **QuestModule** (client): UI panel, progress bars, completion popup
- **QuestConfig**: quest generation with weighted categories
- Categories: CatchFish, CatchRarity, DiscoverSpecies, LootFloatables
- First quest is forced: "Catch 3 Poor+ Fish" with 220g reward (RichText colored label)
- Quest objectives use `incrementObjective(player, objType, amount)` triggered by `FishCaught.OnServerEvent`
- `saveQuest()` calls `store:Set()` to prevent combined blob race conditions

---

## Tutorial System (22 steps)

TutorialModule uses step types: `world` (3D beam), `ui_arrow` (pulse on UI element), `first_slot_arrow` (shop frame icon), `action` (wait for event), `billboard` (3D icon).

Key events fired via `TutorialModule.CompleteEvent(eventName)` from other modules.
Tutorial completion sets `player:SetAttribute("TutorialComplete", true)` and fires `RequestFirstQuest`.

---

## Developer Config & Reset

- `DeveloperConfig`: two-mode system (mode 1 = personal save, mode 2 = dummy data)
- `/resetdata` command: resets all DataStore2 stores, calls `ResetPlayer` on QuestHandler + PlayerMetricsHandler (both re-initialize without rejoin), syncs fresh data to client

---

## Scripts to Provide Claude in New Chats

For most tasks, provide these key files:
1. **EnumModule** — all game enums (always needed)
2. **The specific module(s) you're editing** — full source
3. **ZoneConfig** — if zone/fishing related
4. **RodData + ShipData** — if economy/progression related
5. **QuestConfig + QuestHandler** — if quest related
6. **PlayerMetricsConfig** — if metrics related
7. **LoadGlobals** — if initialization order matters
8. **PlayerJoin** — if data loading/saving is involved

You generally do NOT need to provide: icon modules, OldIcons, ___OLDROOT anything, Zone library internals, WaveModule internals, Socket plugin data.

---

## Common Patterns & Gotchas

- **DataStore2 combined blob race:** Any `store:Set()` triggers a blob save of ALL combined stores. Always save in-memory caches to their store before or alongside other store saves.
- **`Initialize()` vs `PostInitialize()`:** Use PostInitialize when you need `LocalInventory` or other async data.
- **UIManager exclusive screens:** `UIManager.Register(name, openFn, closeFn)` — only one exclusive UI open at a time.
- **Tutorial blocks:** Many systems check `TutorialComplete` attribute or `TutorialModule.IsActive()` to suppress behavior during tutorial.
- **ResetOnSpawn = false:** Major ScreenGuis use this to prevent UI reset on character respawn.
- **Optimistic purchasing:** Client deducts gold + adds item instantly, saves async, verifies async with rollback.
- **`_G.MScript.ModuleName`** pattern: all modules accessed this way, never `require()` directly in game code.
- **Remote naming:** `_G.REvent.EventName` / `_G.RFunc.FuncName` — remotes stored by name, not path.
