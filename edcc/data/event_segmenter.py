"""Event-Aware Segmentation for CAUEEG EEG recordings.

Parses raw event lists from CAUEEG dataset and produces structured segments
with event type labels for EDCC model's event-aware processing.
"""

from enum import IntEnum
from typing import List, Tuple, NamedTuple


class EventType(IntEnum):
    """EDCC event type labels."""
    EYES_OPEN = 0
    EYES_CLOSED = 1
    EO_TO_EC = 2      # Eyes Open → Eyes Closed transition
    EC_TO_EO = 3      # Eyes Closed → Eyes Open transition
    PHOTIC = 4
    OTHER = 5


class Segment(NamedTuple):
    """A labeled segment of an EEG recording."""
    start: int          # start sample index
    end: int            # end sample index (exclusive)
    event_type: int     # EventType value


def _normalize_event_name(name: str) -> str:
    """Normalize raw CAUEEG event name to a canonical category."""
    name = name.strip()
    if name == "Eyes Open":
        return "EO"
    if name == "Eyes Closed":
        return "EC"
    if name.startswith("Photic On"):
        return "PHOTIC_ON"
    if name == "Photic Off":
        return "PHOTIC_OFF"
    return "OTHER"


def parse_events(
    event_list: list,
    signal_length: int,
    transition_margin: int = 600,
    min_segment_length: int = 100,
) -> List[Segment]:
    """Parse a raw CAUEEG event list into labeled segments.

    Args:
        event_list: Raw event list from CauEegDataset, format [[sample_idx, "event_name"], ...].
        signal_length: Total number of samples in the recording.
        transition_margin: Samples before/after a transition boundary to mark as transition (default ±3s at 200Hz).
        min_segment_length: Minimum segment length in samples to keep (default 0.5s at 200Hz).

    Returns:
        List of Segment namedtuples sorted by start time.
    """
    if not event_list:
        return [Segment(0, signal_length, EventType.OTHER)]

    # Extract EO/EC/Photic events with their sample indices
    key_events = []
    for sample_idx, event_name in event_list:
        norm = _normalize_event_name(event_name)
        if norm in ("EO", "EC", "PHOTIC_ON", "PHOTIC_OFF"):
            key_events.append((int(sample_idx), norm))

    if not key_events:
        return [Segment(0, signal_length, EventType.OTHER)]

    # Sort by sample index
    key_events.sort(key=lambda x: x[0])

    # Build raw segments from consecutive events
    raw_segments = []

    # Pre-event region
    if key_events[0][0] > 0:
        raw_segments.append(Segment(0, key_events[0][0], EventType.OTHER))

    for i in range(len(key_events)):
        start_idx = key_events[i][0]
        event_name = key_events[i][1]

        end_idx = key_events[i + 1][0] if i + 1 < len(key_events) else signal_length

        if event_name == "EO":
            etype = EventType.EYES_OPEN
        elif event_name == "EC":
            etype = EventType.EYES_CLOSED
        elif event_name == "PHOTIC_ON":
            etype = EventType.PHOTIC
        elif event_name == "PHOTIC_OFF":
            etype = EventType.OTHER
        else:
            etype = EventType.OTHER

        raw_segments.append(Segment(start_idx, end_idx, etype))

    # Identify EO↔EC transition boundaries and insert transition segments
    transitions = []
    for i in range(len(key_events) - 1):
        curr_name = key_events[i][1]
        next_name = key_events[i + 1][1]
        boundary = key_events[i + 1][0]

        if curr_name == "EO" and next_name == "EC":
            t_start = max(0, boundary - transition_margin)
            t_end = min(signal_length, boundary + transition_margin)
            transitions.append(Segment(t_start, t_end, EventType.EO_TO_EC))
        elif curr_name == "EC" and next_name == "EO":
            t_start = max(0, boundary - transition_margin)
            t_end = min(signal_length, boundary + transition_margin)
            transitions.append(Segment(t_start, t_end, EventType.EC_TO_EO))

    # Merge: raw segments + transitions, giving transitions priority
    all_segments = _merge_with_transitions(raw_segments, transitions, signal_length)

    # Filter short segments
    all_segments = [s for s in all_segments if (s.end - s.start) >= min_segment_length]

    return all_segments


def _merge_with_transitions(
    raw_segments: List[Segment],
    transitions: List[Segment],
    signal_length: int,
) -> List[Segment]:
    """Merge raw segments with transition segments, giving transitions priority.

    Transition segments overwrite the event type of overlapping raw segments.
    """
    if not transitions:
        return raw_segments

    # Create a timeline with event types
    # Use transitions to "punch holes" in raw segments
    result = []

    for seg in raw_segments:
        remaining = [(seg.start, seg.end, seg.event_type)]

        for trans in transitions:
            new_remaining = []
            for r_start, r_end, r_type in remaining:
                # No overlap
                if trans.end <= r_start or trans.start >= r_end:
                    new_remaining.append((r_start, r_end, r_type))
                    continue

                # Left portion before transition
                if r_start < trans.start:
                    new_remaining.append((r_start, trans.start, r_type))

                # Right portion after transition
                if r_end > trans.end:
                    new_remaining.append((trans.end, r_end, r_type))

            remaining = new_remaining

        for r_start, r_end, r_type in remaining:
            result.append(Segment(r_start, r_end, r_type))

    # Add transition segments themselves
    result.extend(transitions)

    # Sort by start time and remove duplicates
    result.sort(key=lambda s: s.start)

    return result


def get_segment_at_sample(segments: List[Segment], sample_idx: int) -> int:
    """Get the event type at a given sample index.

    Args:
        segments: Sorted list of segments.
        sample_idx: Sample index to query.

    Returns:
        EventType value, or EventType.OTHER if not covered.
    """
    for seg in segments:
        if seg.start <= sample_idx < seg.end:
            return seg.event_type
    return EventType.OTHER


def get_window_event_type(
    segments: List[Segment],
    window_start: int,
    window_end: int,
) -> int:
    """Determine the event type for a window based on majority overlap.

    Args:
        segments: Sorted list of segments.
        window_start: Start sample of the window.
        window_end: End sample of the window.

    Returns:
        EventType with the largest overlap with the window.
    """
    overlap_counts = [0] * len(EventType)

    for seg in segments:
        overlap_start = max(window_start, seg.start)
        overlap_end = min(window_end, seg.end)
        if overlap_start < overlap_end:
            overlap_counts[seg.event_type] += overlap_end - overlap_start

    # Prioritize transition types if they have any overlap
    for t in (EventType.EO_TO_EC, EventType.EC_TO_EO):
        if overlap_counts[t] > 0:
            return t

    return int(max(range(len(overlap_counts)), key=lambda i: overlap_counts[i]))


def segment_statistics(segments: List[Segment], sampling_rate: int = 200) -> dict:
    """Compute statistics about segments for debugging/analysis."""
    stats = {etype.name: {"count": 0, "total_seconds": 0.0} for etype in EventType}

    for seg in segments:
        name = EventType(seg.event_type).name
        stats[name]["count"] += 1
        stats[name]["total_seconds"] += (seg.end - seg.start) / sampling_rate

    return stats
