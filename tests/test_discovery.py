import discovery as disc


class TestClassifyDomain:
    def test_plain_company_website_is_website(self):
        assert disc.classify_domain("acme-importers.de") == "website"

    def test_www_prefix_stripped_before_classifying(self):
        assert disc.classify_domain("www.europages.com") == "directory"

    def test_case_insensitive(self):
        assert disc.classify_domain("EUROPAGES.COM") == "directory"

    def test_known_directory_domain(self):
        assert disc.classify_domain("kompass.com") == "directory"

    def test_subdomain_of_directory_domain_still_directory(self):
        assert disc.classify_domain("de.europages.com") == "directory"

    def test_known_report_domain(self):
        assert disc.classify_domain("mordorintelligence.com") == "report"

    def test_known_noise_domain(self):
        assert disc.classify_domain("instagram.com") == "noise"

    def test_facebook_is_social_not_noise(self):
        # facebook.com is deliberately excluded from NOISE_DOMAINS -- smaller
        # importers often run their whole presence as a Facebook Page.
        assert disc.classify_domain("facebook.com") == "social"

    def test_linkedin_is_social(self):
        assert disc.classify_domain("linkedin.com") == "social"

    def test_linkedin_subdomain_is_social(self):
        assert disc.classify_domain("de.linkedin.com") == "social"

    def test_unrelated_domain_is_not_misclassified_as_directory(self):
        # a domain that merely *contains* a known directory name as a
        # substring must not match -- only exact domain or true subdomain.
        assert disc.classify_domain("notkompass.com") == "website"

    def test_lookalike_domain_with_directory_as_prefix_not_subdomain(self):
        assert disc.classify_domain("europages.com.fake-mirror.net") == "website"


class TestMatchesHelper:
    def test_exact_match(self):
        assert disc._matches("kompass.com", {"kompass.com"}) is True

    def test_subdomain_match(self):
        assert disc._matches("eu.kompass.com", {"kompass.com"}) is True

    def test_no_match(self):
        assert disc._matches("acme.com", {"kompass.com"}) is False

    def test_suffix_without_dot_boundary_does_not_match(self):
        # "notkompass.com" ends with "kompass.com" as a raw string suffix,
        # but not on a "." boundary, so it must not be treated as a subdomain.
        assert disc._matches("notkompass.com", {"kompass.com"}) is False


class TestBuildQueries:
    def test_fills_in_product_and_country(self):
        queries = disc.build_queries("Ceramic Tiles", "Germany")
        assert len(queries) == len(disc.QUERY_TEMPLATES)
        assert any("Ceramic Tiles" in q and "Germany" in q for q in queries)

    def test_linkedin_site_query_present(self):
        queries = disc.build_queries("Widgets", "France")
        assert any(q.startswith("site:linkedin.com/company") for q in queries)


class TestCandidateDataclass:
    def test_fields_roundtrip(self):
        c = disc.Candidate(
            url="https://acme.example", domain="acme.example", title="Acme",
            snippet="snippet text", query="acme importers", source_type="website",
        )
        assert c.domain == "acme.example"
        assert c.source_type == "website"
