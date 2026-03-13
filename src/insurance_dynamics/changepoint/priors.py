"""
UK insurance event prior calendar.

Encodes known structural breaks in the UK insurance market as hazard
multipliers for BOCPD. When the algorithm processes a period that coincides
with a known event, the base hazard is multiplied up — making a detected
break more plausible even if the data signal is weak.

This is informative but not dogmatic: if the data show no break around GIPP
implementation, the posterior still won't force one.

Usage
-----
>>> from insurance_changepoint.priors import UKEventPrior
>>> prior = UKEventPrior(lines=['motor'])
>>> hazards = prior.hazard_series(periods, base_hazard=0.01)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np


@dataclass
class UKEvent:
    """A single known UK insurance market event."""

    name: str
    event_date: date
    affected_lines: list[str]
    """e.g. ['motor'], ['property'], ['motor', 'liability'], ['all']"""

    affected_component: str
    """'frequency', 'severity', 'both', or 'pricing'"""

    hazard_multiplier: float
    """How much to multiply the base hazard around this event. Typical: 10–50."""

    confidence: str
    """'hard' (legislation/confirmed), 'soft' (modelled/estimated), or 'uncertain'"""

    window_days: int = 90
    """Number of days either side of event_date where the multiplier applies."""

    notes: str = ""


# The canonical UK event calendar.
# Dates are the effective/announcement dates most relevant to pricing.
UK_EVENTS: list[UKEvent] = [
    UKEvent(
        name="Ogden -0.75%",
        event_date=date(2017, 3, 20),
        affected_lines=["motor", "liability"],
        affected_component="severity",
        hazard_multiplier=40.0,
        confidence="hard",
        window_days=90,
        notes=(
            "Lord Chancellor announced Ogden rate change from 2.5% to -0.75%. "
            "Effective 20 March 2017. Massive step-change in PI reserve costs."
        ),
    ),
    UKEvent(
        name="Ogden -0.25%",
        event_date=date(2019, 8, 5),
        affected_lines=["motor", "liability"],
        affected_component="severity",
        hazard_multiplier=20.0,
        confidence="hard",
        window_days=90,
        notes=(
            "Civil Liability Act 2018 revised Ogden from -0.75% to -0.25%. "
            "Effective 5 August 2019."
        ),
    ),
    UKEvent(
        name="COVID-19 lockdown",
        event_date=date(2020, 3, 23),
        affected_lines=["motor", "liability", "property"],
        affected_component="frequency",
        hazard_multiplier=50.0,
        confidence="hard",
        window_days=60,
        notes=(
            "UK national lockdown from 23 March 2020. Motor frequency collapsed "
            "30-50%. Property frequency mixed (fire up, escape of water down)."
        ),
    ),
    UKEvent(
        name="COVID-19 recovery",
        event_date=date(2020, 6, 15),
        affected_lines=["motor"],
        affected_component="frequency",
        hazard_multiplier=15.0,
        confidence="soft",
        window_days=60,
        notes=(
            "Partial reopening from June 2020. Frequency began recovering toward "
            "normal levels but not fully."
        ),
    ),
    UKEvent(
        name="Storm Ciara/Dennis",
        event_date=date(2020, 2, 9),
        affected_lines=["property"],
        affected_component="severity",
        hazard_multiplier=20.0,
        confidence="hard",
        window_days=45,
        notes=(
            "Storm Ciara 9 Feb 2020, Storm Dennis 16 Feb 2020. Significant "
            "property severity spike. Combined insured losses ~£500m."
        ),
    ),
    UKEvent(
        name="Whiplash Reform",
        event_date=date(2021, 5, 31),
        affected_lines=["motor"],
        affected_component="frequency",
        hazard_multiplier=40.0,
        confidence="hard",
        window_days=90,
        notes=(
            "Civil Liability Act whiplash reforms effective 31 May 2021. "
            "Small claims track limit raised to £5k for RTA PI. "
            "Significant expected reduction in PSLA frequency."
        ),
    ),
    UKEvent(
        name="GIPP implementation",
        event_date=date(2022, 1, 1),
        affected_lines=["motor", "property"],
        affected_component="pricing",
        hazard_multiplier=30.0,
        confidence="hard",
        window_days=90,
        notes=(
            "FCA General Insurance Pricing Practices (PS21/5) effective 1 Jan 2022. "
            "Renewal price cannot exceed equivalent new business price. "
            "Major repricing — effectively a regime break in premium structure."
        ),
    ),
    UKEvent(
        name="Storm Eunice",
        event_date=date(2022, 2, 18),
        affected_lines=["property"],
        affected_component="severity",
        hazard_multiplier=15.0,
        confidence="hard",
        window_days=45,
        notes=(
            "Storm Eunice 18 February 2022. ~£470m insured losses. "
            "Highest wind gusts in 30 years in parts of England."
        ),
    ),
    UKEvent(
        name="Storm Babet",
        event_date=date(2023, 10, 19),
        affected_lines=["property"],
        affected_component="severity",
        hazard_multiplier=15.0,
        confidence="hard",
        window_days=45,
        notes=(
            "Storm Babet 19-21 October 2023. Widespread flooding across "
            "Scotland, northern England. Named storm, ABI-tracked losses."
        ),
    ),
    UKEvent(
        name="Ogden +0.5%",
        event_date=date(2024, 7, 11),
        affected_lines=["motor", "liability"],
        affected_component="severity",
        hazard_multiplier=25.0,
        confidence="hard",
        window_days=180,
        notes=(
            "Lord Chancellor set Ogden rate at +0.5% on 11 July 2024. "
            "Effective for cases resolved from 11 January 2025. "
            "Reduction in BI reserve requirements for long-tail motor."
        ),
    ),
]


class UKEventPrior:
    """
    UK insurance event prior for BOCPD hazard modulation.

    Filters the canonical event calendar to the lines of business relevant
    to the user's portfolio, then computes per-period hazard multipliers.

    Parameters
    ----------
    lines :
        Lines of business to include. Options: 'motor', 'property',
        'liability', 'all'. If 'all' or None, all events are included.
    components :
        Which components to include: 'frequency', 'severity', 'both',
        'pricing', or None for all.
    events :
        Override the event list entirely. If None, uses UK_EVENTS.
    """

    def __init__(
        self,
        lines: list[str] | None = None,
        components: list[str] | None = None,
        events: list[UKEvent] | None = None,
    ) -> None:
        self._all_events = events if events is not None else UK_EVENTS

        self._events = self._filter_events(lines, components)

    def _filter_events(
        self,
        lines: list[str] | None,
        components: list[str] | None,
    ) -> list[UKEvent]:
        filtered = []
        for ev in self._all_events:
            # Line filter
            if lines is not None:
                lines_lower = [l.lower() for l in lines]
                ev_lines = [l.lower() for l in ev.affected_lines]
                if "all" not in ev_lines and not any(
                    l in ev_lines for l in lines_lower
                ):
                    continue
            # Component filter
            if components is not None:
                comp_lower = [c.lower() for c in components]
                if ev.affected_component.lower() not in comp_lower:
                    continue
            filtered.append(ev)
        return filtered

    @property
    def events(self) -> list[UKEvent]:
        """Filtered event list."""
        return self._events

    def hazard_multiplier_for_date(self, d: date) -> float:
        """
        Return the combined hazard multiplier for a given date.

        If multiple events overlap (e.g. COVID lockdown and Storm Ciara
        both affect February 2020 for property), we take the maximum
        multiplier rather than multiplying them together.
        """
        multiplier = 1.0
        for ev in self._events:
            delta = abs((d - ev.event_date).days)
            if delta <= ev.window_days:
                multiplier = max(multiplier, ev.hazard_multiplier)
        return multiplier

    def hazard_series(
        self,
        periods: list[Any],
        base_hazard: float = 0.01,
        period_to_date_fn: Any = None,
        max_hazard: float = 0.5,
    ) -> np.ndarray:
        """
        Compute per-period effective hazard values for a list of periods.

        Parameters
        ----------
        periods :
            List of period labels. If they are ``datetime.date`` objects
            they are used directly. Otherwise ``period_to_date_fn`` must
            be provided.
        base_hazard :
            Base probability of a changepoint per period.
        period_to_date_fn :
            Callable(period) -> datetime.date. Used to convert non-date
            period labels to dates for event matching.
        max_hazard :
            Hard cap on effective hazard (default 0.5 — never force a break).

        Returns
        -------
        np.ndarray of shape (len(periods),) with per-period hazards.
        """
        hazards = np.full(len(periods), base_hazard)

        if not self._events:
            return hazards

        for i, p in enumerate(periods):
            if isinstance(p, date):
                d = p
            elif period_to_date_fn is not None:
                try:
                    d = period_to_date_fn(p)
                except Exception:
                    continue
            else:
                continue

            mult = self.hazard_multiplier_for_date(d)
            hazards[i] = min(base_hazard * mult, max_hazard)

        return hazards

    def summary(self) -> list[dict]:
        """Return list of event dicts for reporting."""
        return [
            {
                "name": ev.name,
                "date": ev.event_date.isoformat(),
                "lines": ev.affected_lines,
                "component": ev.affected_component,
                "multiplier": ev.hazard_multiplier,
                "confidence": ev.confidence,
                "notes": ev.notes,
            }
            for ev in self._events
        ]
