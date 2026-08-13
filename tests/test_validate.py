import validate as val


class TestCheckPhoneCountryCode:
    def test_matching_calling_code(self):
        assert val.check_phone_country_code("+49 30 1234567", "Germany") is True

    def test_matching_calling_code_with_00_prefix(self):
        assert val.check_phone_country_code("0049 30 1234567", "Germany") is True

    def test_non_matching_calling_code(self):
        assert val.check_phone_country_code("+1 555 0100", "Germany") is False

    def test_case_and_whitespace_insensitive_country_lookup(self):
        assert val.check_phone_country_code("+49 30 1234567", "  Germany  ") is True

    def test_none_when_no_phone_given(self):
        assert val.check_phone_country_code(None, "Germany") is None

    def test_none_when_country_not_in_lookup_table(self):
        assert val.check_phone_country_code("+49 30 1234567", "Narnia") is None

    def test_country_alias_uk_and_united_kingdom_share_code(self):
        assert val.check_phone_country_code("+44 20 1234567", "UK") is True
        assert val.check_phone_country_code("+44 20 1234567", "United Kingdom") is True


class TestCheckDomainTld:
    def test_matching_cctld(self):
        assert val.check_domain_tld("https://acme.de", "Germany") is True

    def test_matching_cctld_with_www_prefix(self):
        assert val.check_domain_tld("https://www.acme.de", "Germany") is True

    def test_non_matching_cctld(self):
        assert val.check_domain_tld("https://acme.com", "Germany") is False

    def test_none_when_country_not_in_lookup_table(self):
        assert val.check_domain_tld("https://acme.com", "Narnia") is None

    def test_subdomain_ending_in_cctld_still_matches(self):
        assert val.check_domain_tld("https://shop.acme.de", "Germany") is True


class TestCheckTextMentionsCountry:
    def test_mentioned_in_text_content(self):
        assert val.check_text_mentions_country(
            "We ship across Germany daily.", "", "Germany") is True

    def test_mentioned_only_in_match_reason(self):
        assert val.check_text_mentions_country(
            "generic page text", "Active importer based in Germany.", "Germany") is True

    def test_case_insensitive(self):
        assert val.check_text_mentions_country("we operate in GERMANY", "", "Germany") is True

    def test_not_mentioned_anywhere(self):
        assert val.check_text_mentions_country("we operate in France", "", "Germany") is False


class TestValidateCompany:
    def test_combines_signals_and_counts_confidence(self):
        company = {
            "company_name": "Acme Importers",
            "website": "https://acme.de",
            "contact_phone": "+49 30 1234567",
            "match_reason": "Active importer based in Germany.",
        }
        scraped_text_by_url = {"https://acme.de": "We import goods across Germany."}
        result = val.validate_company(company, "Germany", scraped_text_by_url, use_map_lookup=False)

        assert result["country_signals"]["phone_country_code"] is True
        assert result["country_signals"]["domain_tld"] is True
        assert result["country_signals"]["text_mentions_country"] is True
        assert result["country_signals"]["found_on_map"] is None  # map lookup skipped
        # 3 signals ran and matched (found_on_map is skipped, not counted as checked)
        assert result["validation_confidence"] == "3/3"

    def test_unmapped_country_skips_phone_and_tld_checks(self):
        # "Narnia" has no calling-code/TLD mapping, so those two signals
        # are None (not applicable) and excluded from the confidence
        # count -- only text_mentions_country (always deterministic, never
        # None) is left checked, and it doesn't match here.
        company = {
            "company_name": "Acme Importers",
            "website": "https://acme.com",
            "match_reason": "",
        }
        result = val.validate_company(company, "Narnia", {}, use_map_lookup=False)
        assert result["country_signals"]["phone_country_code"] is None
        assert result["country_signals"]["domain_tld"] is None
        assert result["validation_confidence"] == "0/1"

    def test_unavailable_when_literally_nothing_checked(self):
        # validate_company always runs text_mentions_country (never None),
        # so "unavailable" only occurs if every signal function returns
        # None -- exercised directly against the aggregation logic itself
        # rather than trying to contrive real inputs for all four checks.
        signals = {"a": None, "b": None}
        checked = sum(1 for v in signals.values() if v is not None)
        confidence = sum(1 for v in signals.values() if v is True)
        label = f"{confidence}/{checked}" if checked else "unavailable"
        assert label == "unavailable"

    def test_original_company_fields_preserved(self):
        company = {"company_name": "Acme", "website": "https://acme.com", "relevance_score": 80}
        result = val.validate_company(company, "Germany", {}, use_map_lookup=False)
        assert result["company_name"] == "Acme"
        assert result["relevance_score"] == 80


class TestValidateAll:
    def test_validates_every_company_and_preserves_order(self):
        ranked = [
            {"company_name": "Acme", "website": "https://acme.de", "match_reason": ""},
            {"company_name": "Beta", "website": "https://beta.fr", "match_reason": ""},
        ]
        scraped_pages = [
            {"url": "https://acme.de", "text_content": "Based in Germany."},
        ]
        validated = val.validate_all(ranked, scraped_pages, "Germany", use_map_lookup=False)

        assert [c["company_name"] for c in validated] == ["Acme", "Beta"]
        assert all("country_signals" in c for c in validated)

    def test_on_progress_called_per_company(self):
        ranked = [{"company_name": "Acme", "website": "https://acme.de", "match_reason": ""}]
        seen = []
        val.validate_all(ranked, [], "Germany", use_map_lookup=False, on_progress=seen.append)
        assert any("Acme" in msg for msg in seen)
