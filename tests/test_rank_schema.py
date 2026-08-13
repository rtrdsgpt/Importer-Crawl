import pytest
from pydantic import ValidationError

import rank_schema as rc


def make_judgment(**overrides):
    defaults = dict(
        company_name="Acme Importers GmbH",
        company_role="importer",
        relevance_score=80,
        match_reason="Explicitly describes itself as an importer of the product.",
        contact_email="sales@acme.example",
        contact_phone="+49 30 1234567",
        contact_linkedin="https://linkedin.com/company/acme",
    )
    defaults.update(overrides)
    return rc.LLMJudgment(**defaults)


def make_page(**overrides):
    defaults = dict(
        url="https://acme.example",
        query="widgets importers germany",
        emails=["sales@acme.example"],
        phones=["+49 30 1234567"],
        linkedin_links=["https://linkedin.com/company/acme"],
    )
    defaults.update(overrides)
    return defaults


class TestHallucinationGuardedContacts:
    """to_ranked_company() must never trust a contact value the LLM proposes
    unless the scraper independently found that exact value on the page --
    the whole point of the guard is that a plausible-looking but invented
    contact is worse than no contact at all."""

    def test_contact_kept_when_it_matches_a_scraped_value(self):
        page = make_page()
        judgment = make_judgment()
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_email == "sales@acme.example"
        assert result.contact_phone == "+49 30 1234567"
        assert result.contact_linkedin == "https://linkedin.com/company/acme"

    def test_hallucinated_email_falls_back_to_first_scraped_email(self):
        page = make_page(emails=["real@acme.example", "other@acme.example"])
        judgment = make_judgment(contact_email="invented@fake.example")
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_email == "real@acme.example"

    def test_hallucinated_phone_falls_back_to_first_scraped_phone(self):
        page = make_page(phones=["+49 30 1111111"])
        judgment = make_judgment(contact_phone="+1 555 0100")
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_phone == "+49 30 1111111"

    def test_hallucinated_linkedin_falls_back_to_first_scraped_linkedin(self):
        page = make_page(linkedin_links=["https://linkedin.com/company/real"])
        judgment = make_judgment(contact_linkedin="https://linkedin.com/company/fake")
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_linkedin == "https://linkedin.com/company/real"

    def test_no_scraped_contacts_means_none_even_if_llm_proposes_one(self):
        page = make_page(emails=[], phones=[], linkedin_links=[])
        judgment = make_judgment(
            contact_email="invented@fake.example",
            contact_phone="+1 555 0100",
            contact_linkedin="https://linkedin.com/company/fake",
        )
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_email is None
        assert result.contact_phone is None
        assert result.contact_linkedin is None

    def test_llm_proposes_none_but_scraper_found_a_value_falls_back_to_it(self):
        page = make_page(emails=["found@acme.example"])
        judgment = make_judgment(contact_email=None)
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_email == "found@acme.example"

    def test_missing_emails_key_treated_as_empty(self):
        page = make_page()
        del page["emails"]
        judgment = make_judgment()
        result = rc.to_ranked_company(page, judgment)
        assert result.contact_email is None

    def test_sources_used_and_role_preserved(self):
        page = make_page(url="https://acme.example/about", query="acme importers")
        judgment = make_judgment(company_role="distributor")
        result = rc.to_ranked_company(page, judgment)
        assert result.sources_used == ["https://acme.example/about", "acme importers"]
        assert result.company_role == "distributor"
        assert result.website == "https://acme.example/about"


class TestLLMJudgmentListCoercion:
    """Some models return a JSON array for a contact field instead of a
    single string when a page has multiple candidates -- _first_if_list
    should coerce that to the first non-empty value rather than fail."""

    def test_list_email_coerced_to_first_value(self):
        judgment = rc.LLMJudgment(
            company_name="Acme",
            company_role="importer",
            relevance_score=50,
            match_reason="reason",
            contact_email=["first@acme.example", "second@acme.example"],
        )
        assert judgment.contact_email == "first@acme.example"

    def test_list_with_falsy_entries_skips_them(self):
        judgment = rc.LLMJudgment(
            company_name="Acme",
            company_role="importer",
            relevance_score=50,
            match_reason="reason",
            contact_phone=["", None, "+49 30 1234567"],
        )
        assert judgment.contact_phone == "+49 30 1234567"

    def test_plain_string_passes_through_unchanged(self):
        judgment = rc.LLMJudgment(
            company_name="Acme",
            company_role="importer",
            relevance_score=50,
            match_reason="reason",
            contact_email="plain@acme.example",
        )
        assert judgment.contact_email == "plain@acme.example"

    def test_relevance_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            rc.LLMJudgment(
                company_name="Acme", company_role="importer",
                relevance_score=150, match_reason="reason",
            )

    def test_invalid_company_role_rejected(self):
        with pytest.raises(ValidationError):
            rc.LLMJudgment(
                company_name="Acme", company_role="seller",
                relevance_score=50, match_reason="reason",
            )


class TestEligiblePages:
    def test_keeps_scraped_websites_and_social_snippets(self):
        pages = [
            {"status": "success", "source_type": "website"},
            {"status": "snippet_only", "source_type": "social"},
        ]
        assert rc.eligible_pages(pages) == pages

    def test_excludes_directory_pages(self):
        pages = [{"status": "success", "source_type": "directory"}]
        assert rc.eligible_pages(pages) == []

    def test_excludes_failed_or_noise_pages(self):
        pages = [
            {"status": "failed", "source_type": "website"},
            {"status": "success", "source_type": "noise"},
            {"status": "skipped_noise", "source_type": "noise"},
        ]
        assert rc.eligible_pages(pages) == []


class TestExtractJsonObject:
    def test_plain_json(self):
        assert rc.extract_json_object('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self):
        text = '```json\n{"a": 1}\n```'
        assert rc.extract_json_object(text) == {"a": 1}

    def test_extracts_object_from_surrounding_prose(self):
        text = 'Sure, here you go:\n{"a": 1}\nHope that helps!'
        assert rc.extract_json_object(text) == {"a": 1}

    def test_raises_on_empty_text(self):
        with pytest.raises(ValueError):
            rc.extract_json_object(None)
        with pytest.raises(ValueError):
            rc.extract_json_object("")

    def test_raises_when_no_json_object_present(self):
        with pytest.raises(ValueError):
            rc.extract_json_object("no json here")


class TestFirstSentence:
    def test_extracts_first_sentence(self):
        assert rc.first_sentence("First one. Second one.") == "First one."

    def test_falls_back_to_full_text_when_no_terminator(self):
        assert rc.first_sentence("no terminator here") == "no terminator here"
