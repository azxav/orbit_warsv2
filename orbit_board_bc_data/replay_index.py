from __future__ import annotations

from .schema import Fleet, Planet

FleetRow = list[float]
PlanetRow = list[float]


class IndexedReplayFrames:
    """Per-player replay observations with cached entity lookups."""

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames
        self._raw_fleet_by_id: list[dict[int, list[float]]] = []
        self._fleet_cache: list[dict[int, Fleet]] = []
        self._fleet_list_cache: list[list[Fleet] | None] = []
        self._raw_planets: list[list[list[float]]] = []
        self._planet_cache: list[list[Planet] | None] = []
        self.fleet_lookup_count = 0
        for obs in frames:
            self._raw_fleet_by_id.append({int(raw[0]): raw for raw in obs.get("fleets", [])})
            self._fleet_cache.append({})
            self._fleet_list_cache.append(None)
            self._raw_planets.append(list(obs.get("planets", [])))
            self._planet_cache.append(None)

    def __len__(self) -> int:
        return len(self._frames)

    def frame(self, index: int) -> dict:
        return self._frames[index]

    def fleet_by_id(self, index: int, fleet_id: int) -> Fleet | None:
        self.fleet_lookup_count += 1
        normalized_id = int(fleet_id)
        cached = self._fleet_cache[index].get(normalized_id)
        if cached is not None:
            return cached
        raw = self._raw_fleet_by_id[index].get(normalized_id)
        if raw is None:
            return None
        fleet = Fleet.from_raw(raw)
        self._fleet_cache[index][normalized_id] = fleet
        return fleet

    def fleet_row_by_id(self, index: int, fleet_id: int) -> FleetRow | None:
        self.fleet_lookup_count += 1
        return self._raw_fleet_by_id[index].get(int(fleet_id))

    def fleet_ids(self, index: int) -> set[int]:
        return set(self._raw_fleet_by_id[index])

    def fleet_rows(self, index: int) -> list[FleetRow]:
        return list(self._raw_fleet_by_id[index].values())

    def fleets(self, index: int) -> list[Fleet]:
        cached = self._fleet_list_cache[index]
        if cached is None:
            cached = []
            for fleet_id, raw in self._raw_fleet_by_id[index].items():
                fleet = Fleet.from_raw(raw)
                self._fleet_cache[index][fleet_id] = fleet
                cached.append(fleet)
            self._fleet_list_cache[index] = cached
        return self._fleet_list_cache[index] or []

    def planets(self, index: int) -> list[Planet]:
        cached = self._planet_cache[index]
        if cached is None:
            cached = [Planet.from_raw(raw) for raw in self._raw_planets[index]]
            self._planet_cache[index] = cached
        return cached

    def planet_rows(self, index: int) -> list[PlanetRow]:
        return self._raw_planets[index]
