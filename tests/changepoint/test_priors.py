"""
Tests for UKEventPrior and the UK event calendar.
"""

import numpy as np
import pytest
from datetime import date, timedelta

from insurance_dynamics.changepoint.priors import UKEventPrior, UK_EVENTS, UKEvent


class TestUKEventCalendar:
    def test_event_count(self):
        """Calendar should have at least 10 events."""
        assert len(UK_EVENTS) >= 10

    def test_all_events_have_required_fields(self):
        for ev in UK_EVENTS:
            assert ev.name
            assert isinstance(ev.event_date, date)
            assert ev.affected_lines
            assert ev.affected_component in (
                "frequency", "severity", "both", "pricing"
            )
            assert ev.hazard_multiplier > 1.0
            assert ev.confidence in ("hard", "soft", "uncertain")

    def test_key_events_present(self):
        names = {ev.name for ev in UK_EVENTS}
        assert any("Ogden" in n for n in names)
        assert any("COVID" in n for n in names)
        assert any("Whiplash" in n for n in names)
        assert any("GIPP" in n for n in names)

    def test_ogden_2017_date(self):
        ogden_2017 = next(
            ev for ev in UK_EVENTS if "Ogden" in ev.name and "0.75" in ev.name
        )
        assert ogden_2017.event_date.year == 2017

    def test_covid_lockdown_date(self):
        covid = next(
            ev for ev in UK_EVENTS
            if "COVID" in ev.name and "lockdown" in ev.name.lower()
        )
        assert covid.event_date == date(2020, 3, 23)
        assert covid.hazard_multiplier >= 40


class TestUKEventPrior:
    def test_no_filter_includes_all_events(self):
        prior = UKEventPrior()
        assert len(prior.events) == len(UK_EVENTS)

    def test_motor_filter(self):
        prior = UKEventPrior(lines=["motor"])
        # All filtered events should affect motor
        for ev in prior.events:
            assert any(
                l in ["motor", "all"] for l in [x.lower() for x in ev.affected_lines]
            )

    def test_property_filter(self):
        prior = UKEventPrior(lines=["property"])
        for ev in prior.events:
            assert any(
                l in ["property", "all"] for l in [x.lower() for x in ev.affected_lines]
            )

    def test_component_filter_frequency(self):
        prior = UKEventPrior(components=["frequency"])
        for ev in prior.events:
            assert ev.affected_component == "frequency"

    def test_component_filter_severity(self):
        prior = UKEventPrior(components=["severity"])
        for ev in prior.events:
            assert ev.affected_component == "severity"

    def test_multiplier_for_covid_date(self):
        prior = UKEventPrior(lines=["motor"], components=["frequency"])
        mult = prior.hazard_multiplier_for_date(date(2020, 3, 23))
        # COVID lockdown should give high multiplier
        assert mult >= 10.0

    def test_multiplier_for_normal_date(self):
        prior = UKEventPrior()
        # A random date with no event should return 1.0
        mult = prior.hazard_multiplier_for_date(date(2018, 6, 15))
        assert mult == 1.0

    def test_hazard_series_shape(self):
        prior = UKEventPrior()
        periods = [date(2020, m, 1) for m in range(1, 13)]
        hazards = prior.hazard_series(periods, base_hazard=0.01)
        assert hazards.shape == (12,)

    def test_hazard_series_base_hazard_respected(self):
        prior = UKEventPrior()
        # Date far from any event
        periods = [date(2018, 6, 1)]
        hazards = prior.hazard_series(periods, base_hazard=0.02)
        assert np.isclose(hazards[0], 0.02)

    def test_hazard_series_event_period_elevated(self):
        prior = UKEventPrior(lines=["motor"], components=["frequency"])
        # COVID lockdown period
        periods = [date(2019, 1, 1), date(2020, 3, 23), date(2021, 1, 1)]
        hazards = prior.hazard_series(periods, base_hazard=0.01)
        # COVID period should be highest
        assert hazards[1] > hazards[0]
        assert hazards[1] > hazards[2]

    def test_hazard_series_max_hazard_cap(self):
        prior = UKEventPrior()
        periods = [date(2020, 3, 23)]
        hazards = prior.hazard_series(periods, base_hazard=0.01, max_hazard=0.3)
        assert hazards[0] <= 0.3

    def test_hazard_series_non_date_periods_without_fn(self):
        """Non-date periods without a conversion function should return base hazard."""
        prior = UKEventPrior()
        periods = ["2020-Q1", "2020-Q2", "2020-Q3"]
        hazards = prior.hazard_series(periods, base_hazard=0.01)
        # Should not raise; periods not matched = base hazard returned
        assert np.allclose(hazards, 0.01)

    def test_hazard_series_with_period_to_date_fn(self):
        """Provide a conversion function for non-date periods."""
        from datetime import date

        def q_to_date(q: str) -> date:
            year, quarter = q.split("-Q")
            month = (int(quarter) - 1) * 3 + 1
            return date(int(year), month, 1)

        prior = UKEventPrior(lines=["motor"], components=["frequency"])
        periods = ["2019-Q4", "2020-Q1", "2020-Q2", "2021-Q1"]
        hazards = prior.hazard_series(
            periods, base_hazard=0.01, period_to_date_fn=q_to_date
        )
        # 2020-Q2 = Apr 1 2020, within COVID lockdown 60-day window (9 days after)
        assert hazards[2] > hazards[0]

    def test_summary_returns_list_of_dicts(self):
        prior = UKEventPrior()
        summary = prior.summary()
        assert isinstance(summary, list)
        assert len(summary) > 0
        assert "name" in summary[0]
        assert "date" in summary[0]
        assert "multiplier" in summary[0]

    def test_custom_events(self):
        custom = [
            UKEvent(
                name="Test Event",
                event_date=date(2023, 6, 1),
                affected_lines=["motor"],
                affected_component="frequency",
                hazard_multiplier=20.0,
                confidence="soft",
            )
        ]
        prior = UKEventPrior(events=custom)
        assert len(prior.events) == 1
        mult = prior.hazard_multiplier_for_date(date(2023, 6, 1))
        assert mult == 20.0

    def test_overlapping_events_takes_maximum(self):
        """Overlapping events: multiplier should be max, not product."""
        custom = [
            UKEvent(
                name="Event A",
                event_date=date(2020, 3, 1),
                affected_lines=["motor"],
                affected_component="frequency",
                hazard_multiplier=10.0,
                confidence="hard",
                window_days=30,
            ),
            UKEvent(
                name="Event B",
                event_date=date(2020, 3, 15),
                affected_lines=["motor"],
                affected_component="frequency",
                hazard_multiplier=25.0,
                confidence="hard",
                window_days=30,
            ),
        ]
        prior = UKEventPrior(events=custom)
        mult = prior.hazard_multiplier_for_date(date(2020, 3, 10))
        assert mult == 25.0  # max, not 250
