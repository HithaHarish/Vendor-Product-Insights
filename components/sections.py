import streamlit as st
def render_intro_sections():
    # ---------------------------------------------------
    # What We Do
    # ---------------------------------------------------
    st.markdown("""
    <div class="section-heading">What We Do?</div>
    <div class="section-underline"></div>

    <p class="section-text">
    This platform delivers <span class="highlight">genuine review intelligence</span> to help vendors
    improve product quality and protect marketplace trust. It separates suspicious feedback from reliable
    customer voice, then converts trusted review signals into <span class="highlight">clear product improvement insights</span>.
    The result is a practical view of what customers truly like, what needs improvement, and how fake reviews
    distort perceived product performance.
    </p>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------
    # How We Do It
    # ---------------------------------------------------
    st.markdown("""
    <div class="section-heading">How We Do It?</div>
    <div class="section-underline"></div>

    <ul class="section-text custom-list">

    <li>
    <span class="highlight-core">Multi-Layer Review Intelligence :</span>
    The system combines <span class="highlight-core">textual</span>, <span class="highlight-core">behavioral</span>,
    and <span class="highlight-core">temporal</span> analysis to identify suspicious patterns and isolate genuine feedback.
    This ensures product decisions are based on authentic customer signals.
    </li>

    <li>
    <span class="highlight-core">Genuine-Only Product Insights :</span>
    After fake-review removal, the platform builds trustworthy metrics such as corrected rating,
    sentiment distribution, feature-level strengths/weaknesses, and improvement impact ranking.
    These insights guide vendors toward high-impact quality improvements.
    </li>

    <li>
    <span class="highlight-core">Vendor Decision Support :</span>
    The dashboard compares raw vs genuine outcomes, quantifies rating inflation, and generates
    risk alerts for products with high fraud pressure. This helps teams prioritize moderation,
    safeguard trust, and improve product strategy with evidence-backed signals.
    </li>

    </ul>
    """, unsafe_allow_html=True)
