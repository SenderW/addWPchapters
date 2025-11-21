#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WP AI-Impact Inserter v14.6

This script connects to a WordPress site and automatically inserts an additional expert section into
selected posts on a WordPress site via the REST API. It uses the DeepSeek chat completion API to
generate French language content that extends the existing article with current developments and
context, for example an extra block with a heading and several paragraphs related to a cross topic
you define for your own use case.

The script is intended to be run from the command line and controlled via environment variables.
It is safe to publish this file on GitHub as long as all credentials are provided only through
environment variables and not hard coded in the source code.
"""

import os
import sys
import re
import html
import json
import time
from typing import Tuple, Optional, Any, Dict, List, Set
from urllib.parse import urlsplit
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# Configuration
# =========================

@dataclass
class Config:
    """Central configuration for WordPress and AI access"""
    wp_base: str = os.environ.get("WP_BASE", "https://example.com").rstrip("/")
    wp_user: str = os.environ.get("WP_USER", "").strip()
    wp_app_password: str = os.environ.get("WP_APP_PASSWORD", "").strip()
    
    wp_cookie: str = os.environ.get("WP_COOKIE", "").strip()
    wp_nonce: str = os.environ.get("WP_NONCE", "").strip()
    
    wordpress_a_url: str = os.environ.get("WORDPRESS_A_URL", "https://example.com").rstrip("/")
    wordpress_a_username: str = os.environ.get("WORDPRESS_A_USERNAME", "").strip()
    wordpress_a_password: str = os.environ.get("WORDPRESS_A_PASSWORD", "").strip()
    
    deepseek_api_key: str = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "").strip()
    
    per_page: int = 10
    statuses: List[str] = None
    request_timeout: Tuple[float, float] = (10.0, 120.0)
    sleep_between: float = float(os.environ.get("SLEEP_BETWEEN", "0.4"))
    max_tokens: int = 900
    model: str = "deepseek-chat"
    
    only_title_contains: str = "AI-Impact"
    allowed_fields: Tuple[str, ...] = ("content",)
    skip_categories: Tuple[int, ...] = ()
    min_content_length: int = 600
    
    auto_continue: bool = os.environ.get("AUTO_CONTINUE", "").strip().lower() in ("1", "true", "yes", "y")
    auto_quit: bool = os.environ.get("AUTO_QUIT", "").strip().lower() in ("1", "true", "yes", "y")
    auto_all: bool = os.environ.get("AUTO_ALL", "").strip().lower() in ("1", "true", "yes", "y")
    
    processed_log_path: str = os.environ.get("PROCESSED_LOG_PATH", ".processed_posts").strip() or ".processed_posts"
    
    # Optional skiplists. If empty, skiplist loading is disabled.
    skiplist_url_1: str = os.environ.get("SKIPLIST_URL_1", "").strip()
    skiplist_url_2: str = os.environ.get("SKIPLIST_URL_2", "").strip()
    
    def __post_init__(self):
        if self.statuses is None:
            self.statuses = ["publish"]

# Colors
BLUE = "\x1b[34m"
GREEN = "\033[92m"
RESET = "\033[0m"
YELLOW = "\033[93m"
RED = "\033[91m"

PROHIBITED = (
    "vital, nestled, uncover, journey, embark, unleash, dive, delve, discover, "
    "plethora, indulge, more than just, not just, unlock, unveil, look no further, "
    "world of, realm, elevate, whether you're, landscape, navigate, daunting, both style, "
    "tapestry, unique blend, blend, enhancing, game changer, stand out, stark, contrast"
)

CONFIG = Config()

# =========================
# Helpers
# =========================

def create_session(config: Config) -> requests.Session:
    """Create configured Session with retry logic"""
    session = requests.Session()
    
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.request_timeout = config.request_timeout  # type: ignore[attr-defined]
    return session


def strip_html_to_text(text: str) -> str:
    """Remove HTML and WP blocks and return plain text"""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[/?[a-zA-Z0-9_-]+[^\]]*\]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_plaintext(s: str) -> str:
    """Clean text for API requests"""
    s = s.replace('"', "'")
    s = s.replace("“", "'").replace("”", "'")
    s = s.replace("„", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_url(u: str) -> str:
    """Normalize URL without trailing slash"""
    u = u.strip()
    if not u:
        return u
    parts = urlsplit(u)
    path = parts.path.rstrip("/") or "/"
    return f"{parts.scheme}://{parts.netloc}{path}"


def remove_trailing_period(s: str) -> str:
    """Remove period at the end"""
    return s[:-1] if s.endswith(".") else s


# =========================
# Processed Posts Manager
# =========================

class ProcessedPostsManager:
    """Manage list of processed posts"""
    
    def __init__(self, path: str):
        self.log_path = path
        self.json_path = path + ".json"
    
    def load(self) -> Set[int]:
        """Load processed post IDs"""
        ids = set()
        ids.update(self._read_file(self.log_path))
        ids.update(self._read_file(self.json_path))
        return ids
    
    def _read_file(self, path: str) -> Set[int]:
        """Read IDs from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if not raw:
                    return set()
            
            if raw.lstrip().startswith("{"):
                data = json.loads(raw)
                return set(map(int, data.get("post_ids", [])))
            
            ids = set()
            for line in raw.splitlines():
                line = line.strip()
                if line and line.isdigit():
                    ids.add(int(line))
            return ids
            
        except FileNotFoundError:
            return set()
        except Exception as e:
            print(f"{YELLOW}[WARN]{RESET} Error reading {path}: {e}")
            return set()
    
    def save(self, ids: Set[int]) -> None:
        """Save IDs in both formats"""
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                for pid in sorted(ids):
                    f.write(f"{pid}\n")
        except Exception as e:
            print(f"{YELLOW}[WARN]{RESET} Error writing {self.log_path}: {e}")
        
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({"post_ids": sorted(ids)}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def mark(self, post_id: int) -> None:
        """Mark post as processed"""
        ids = self.load()
        ids.add(int(post_id))
        self.save(ids)


# =========================
# WordPress Endpoints
# =========================

def build_endpoints(base: str) -> Dict[str, str]:
    """Create API endpoints"""
    base = base.rstrip("/")
    return {
        "posts": f"{base}/wp-json/wp/v2/posts",
        "media": f"{base}/wp-json/wp/v2/media",
        "user": f"{base}/wp-json/wp/v2/users/me",
        "rest_lock": f"{base}/wp-json/wp/v2/settings",
    }


# =========================
# Auth Handling
# =========================

class AuthManager:
    """Manage WordPress authentication"""
    
    def __init__(self, config: Config, session: requests.Session):
        self.config = config
        self.session = session
        self.auth = None
        self.cookie_headers = {}
        self.mode = "basic"
        self.active_base = config.wp_base
        self.endpoints = build_endpoints(self.active_base)
    
    def initialize(self) -> str:
        """Initialize auth and return active base URL"""
        if self.config.wp_user and self.config.wp_app_password:
            if self._try_basic_auth(self.config.wp_base, self.config.wp_user, self.config.wp_app_password):
                self.mode = "basic"
                return self.active_base
        
        if self.config.wordpress_a_username and self.config.wordpress_a_password:
            if self._try_basic_auth(
                self.config.wordpress_a_url,
                self.config.wordpress_a_username,
                self.config.wordpress_a_password,
            ):
                self.mode = "basic"
                self.active_base = self.config.wordpress_a_url
                self.endpoints = build_endpoints(self.active_base)
                return self.active_base
        
        if self.config.wp_cookie and self.config.wp_nonce:
            if self._try_cookie_nonce():
                self.mode = "cookie"
                return self.active_base
        
        raise RuntimeError("Authentication failed for all configured WordPress endpoints")
    
    def _try_basic_auth(self, base: str, user: str, password: str) -> bool:
        """Try basic auth"""
        endpoints = build_endpoints(base)
        self.session.headers.update({"Accept": "application/json"})
        self.auth = (user, password)
        
        try:
            resp = self.session.get(
                endpoints["user"],
                auth=self.auth,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            print(
                f"{GREEN}[AUTH]{RESET} Basic auth on {base} | User: {data.get('name')} | Roles: {data.get('roles')}"
            )
            self.endpoints = endpoints
            self.active_base = base
            return True
        except Exception as e:
            print(f"{YELLOW}[INFO]{RESET} Basic Auth on {base} failed: {e}")
            self.auth = None
            return False
    
    def _try_cookie_nonce(self) -> bool:
        """Try cookie plus nonce auth"""
        headers = {
            "X-WP-Nonce": self.config.wp_nonce,
            "Cookie": self.config.wp_cookie,
            "Accept": "application/json",
        }
        try:
            resp = self.session.get(
                self.endpoints["user"],
                headers=headers,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            print(
                f"{GREEN}[AUTH]{RESET} Cookie/Nonce | User: {data.get('name')} | Roles: {data.get('roles')}"
            )
            self.cookie_headers = headers
            self.mode = "cookie"
            return True
        except Exception as e:
            print(f"{YELLOW}[INFO]{RESET} Cookie/Nonce auth failed: {e}")
            self.cookie_headers = {}
            return False
    
    def _clear_cookie_headers(self):
        """Remove cookie and nonce headers from session"""
        for key in ["X-WP-Nonce", "Cookie"]:
            self.session.headers.pop(key, None)
    
    def get_auth(self):
        """Return auth object for requests"""
        if self.mode == "cookie":
            return None
        return self.auth


# =========================
# WordPress Client
# =========================

class WordPressClient:
    """Simple REST client for WordPress"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = create_session(config)
        self.auth_mgr = AuthManager(config, self.session)
        self.auth_mgr.initialize()
        self.session.headers.update({"Accept": "application/json"})
    
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        auth = self.auth_mgr.get_auth()
        headers = kwargs.pop("headers", {})
        headers.update(self.auth_mgr.cookie_headers)
        
        resp = self.session.request(
            method,
            url,
            headers=headers,
            auth=auth,
            timeout=self.config.request_timeout,
            **kwargs,
        )
        resp.raise_for_status()
        return resp
    
    def get_user_info(self) -> Dict[str, Any]:
        try:
            resp = self._request("GET", self.auth_mgr.endpoints["user"])
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"{YELLOW}[WARN]{RESET} user info: {e}\n")
            return {}
    
    def get_posts(
        self,
        page: int,
        per_page: int,
        statuses: List[str],
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params = {"page": page, "per_page": per_page, "status": ",".join(statuses)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        resp = self._request("GET", self.auth_mgr.endpoints["posts"], params=params)
        return resp.json()
    
    def update_post(self, post_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.auth_mgr.endpoints['posts']}/{post_id}"
        resp = self._request("PUT", url, json=data)
        return resp.json()
    
    def get_post(self, post_id: int) -> Dict[str, Any]:
        url = f"{self.auth_mgr.endpoints['posts']}/{post_id}"
        resp = self._request("GET", url)
        return resp.json()
    
    def get_media(self, media_id: int) -> Optional[Dict[str, Any]]:
        if not media_id:
            return None
        url = f"{self.auth_mgr.endpoints['media']}/{media_id}"
        resp = self._request("GET", url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    
    def update_media_alt(self, media_id: int, alt_text: str) -> None:
        """Update media alt text"""
        url = f"{self.auth_mgr.endpoints['media']}/{media_id}"
        try:
            resp = self._request("PUT", url, json={"alt_text": alt_text})
            resp.raise_for_status()
        except Exception as e:
            print(f"{YELLOW}[WARN]{RESET} Media alt update failed: {e}")
    
    def load_skiplist(self, url: str) -> List[str]:
        """Load skiplist from URL"""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return []
            return [line.strip() for line in text.splitlines() if line.strip()]
        except Exception as e:
            print(f"{YELLOW}[WARN]{RESET} Skiplist load from {url} failed: {e}")
            return []
    
    def load_skiplist_urls(self, url: str) -> Set[str]:
        """Load URL skiplist and normalize entries"""
        raw = self.load_skiplist(url)
        return {normalize_url(u) for u in raw}


# =========================
# DeepSeek API
# =========================

def call_deepseek(
    config: Config,
    messages: List[Dict[str, str]],
    max_tokens: int = 400,
    temperature: float = 0.7,
) -> str:
    """Call DeepSeek API"""
    if not config.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    return data["choices"][0]["message"]["content"].strip()


# =========================
# Domain logic: Filters and AI content
# =========================

def is_relevant_post(title: str, text: str) -> bool:
    """
Hook to decide if a post is relevant for processing.

Default implementation returns True for all posts. You can customize this function
for your own use case, for example by checking for certain keywords or categories.
"""
    return True


def has_ai_coverage(title: str, text: str) -> bool:
    """
Simple example check whether a cross topic (here AI related terms) already appears in the text.

This is a safeguard to avoid inserting a second section if the article already contains
a detailed discussion of the same cross topic.
"""
    plain = sanitize_plaintext(strip_html_to_text(title + " " + text)).lower()
    ai_keywords = [
        "künstliche intelligenz",
        "ki",
        "artificial intelligence",
        "ai",
    ]
    
    has_ai = any(k in plain for k in ai_keywords)
    if not has_ai:
        return False
    
    count_ai = sum(plain.count(k) for k in ai_keywords)
    return count_ai >= 3


def generate_area_heading(title: str, text: str) -> str:
    """Generate H2 heading like a cross topic label for the field X"""
    prompt = f"""
Tu es un expert des contenus spécialisés.

Sur la base du titre et d'un extrait d'un article, tu dois trouver une désignation de domaine précise,
claire et courte qui décrit le centre d'intérêt principal de l'article.

Titre: "{sanitize_plaintext(title)}"
Texte (extrait): "{sanitize_plaintext(strip_html_to_text(text)[:800])}"

1. Détermine le thème principal:
   - Il peut s'agir d'un sujet technique, d'un secteur, d'un thème de société ou d'un autre domaine spécialisé.
   - Évite des formulations trop générales.
   - Concentre toi sur le dénominateur commun le plus concret de l'article.

2. Formule une désignation X qui:
   - est courte (2 à 5 mots maximum)
   - peut être au pluriel ou comme terme générique
   - n'est pas une phrase entière mais seulement un nom de domaine.

3. Donne le résultat uniquement sous la forme suivante:
   Sous thème pour le domaine X

Tu remplaces X par ta désignation, par exemple:
   Sous thème pour le domaine technologies numériques
   Sous thème pour le domaine gestion de projets
   Sous thème pour le domaine communication en ligne

Règles:
- Reste proche du thème principal de l'article.
- N'ajoute pas de texte avant ou après cette phrase.
"""
    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant précis pour l'analyse de textes spécialisés.",
        },
        {"role": "user", "content": prompt},
    ]
    heading = call_deepseek(CONFIG, messages, max_tokens=80, temperature=0.4)
    heading = heading.strip().replace("\n", " ")
    heading = remove_trailing_period(heading)
    if not heading.lower().startswith("sous thème pour le domaine"):
        heading = f"Sous thème pour le domaine {heading}"
    return heading


def normalize_heading_with_deepseek(heading: str, title: str, text: str) -> str:
    """
Normalise ce sous titre H2 pour un article français sur un domaine spécialisé:

"{heading}"

Utilise le titre et un extrait de l'article:

Titre: "{sanitize_plaintext(title)}"
Texte (extrait): "{sanitize_plaintext(strip_html_to_text(text)[:800])}"

Objectif: une H2 exactement au format suivant:

Sous thème pour le domaine X

Règles:
- X contient au maximum 5 mots.
- X n'est pas une phrase entière mais une désignation comme:
  - "technologies numériques"
  - "gestion de projets"
  - "communication scientifique"
- X doit refléter le cœur du contenu de l'article.

Donne uniquement la H2 complète dans cette forme, sans explications supplémentaires:

Sous thème pour le domaine X
"""
    messages = [
        {
            "role": "system",
            "content": "Tu formules des sous titres H2 précis et courts en français.",
        },
        {"role": "user", "content": normalize_heading_with_deepseek.__doc__},
    ]
    result = call_deepseek(CONFIG, messages, max_tokens=60, temperature=0.3)
    result = result.strip().replace("\n", " ")
    result = remove_trailing_period(result)
    if not result.lower().startswith("sous thème pour le domaine"):
        result = heading
    return result


def choose_sender_designation(title: str, text: str) -> str:
    """
Return a generic expert designation in French.

You can replace the returned string with a project specific label, for example
"expert en communication", "spécialiste du sujet", etc.
"""
    return "expert du sujet"


def generate_opening_question(area_label: str) -> str:
    """Generate opening question in French for the cross topic section"""
    prompt = f"""
Tu es journaliste spécialisé. Tu dois formuler une question d'ouverture concise pour un paragraphe
qui traite d'un sous thème dans un certain domaine.

Désignation du domaine:
"{sanitize_plaintext(area_label)}"

Tâche:
- Formule une seule question en français qui donne envie de lire la suite.
- La question s'adresse à des personnes qui s'intéressent à ce domaine.
- La question doit être donnée sans guillemets.

Exemples de forme (à adapter, ne pas copier mot à mot):
- "Comment ce domaine évolue t il actuellement pour les personnes concernées?"
- "Quelles nouvelles exigences apparaissent progressivement dans ce domaine?"
- "Quels changements pratiques peuvent toucher le quotidien dans ce domaine?"

Règles:
- Intègre la désignation du domaine de manière naturelle.
- Reste factuel mais clair.
- Donne uniquement la question, sans texte supplémentaire.
"""
    messages = [
        {"role": "system", "content": "Tu formules des questions d'ouverture précises en français."},
        {"role": "user", "content": prompt},
    ]
    q = call_deepseek(CONFIG, messages, max_tokens=60, temperature=0.7)
    q = remove_trailing_period(q.strip().replace("\n", " "))
    if not q.endswith("?"):
        q += "?"
    return q


def generate_ai_impact_paragraphs(
    title: str,
    text: str,
    sender_role: str,
    area_label: str,
) -> List[str]:
    """
Generate paragraphs for the additional expert section.

French content, 3 paragraphs, with a short closing hint to changing requirements.
"""
    article_excerpt = sanitize_plaintext(strip_html_to_text(text)[:1200])
    prompt = f"""
Tu es {sender_role}. Tu rédiges une courte analyse en français.

Contexte:
- Article pour le domaine: "{sanitize_plaintext(area_label)}"
- Titre: "{sanitize_plaintext(title)}"
- Extrait du texte: "{article_excerpt}"
- Public cible: personnes intéressées par ce domaine.

Tâche:
- Écris un commentaire concis (3 paragraphes) qui:
  - met en évidence des évolutions actuelles pertinentes,
  - décrit des opportunités possibles,
  - indique aussi des risques ou des incertitudes,
  - explique comment les exigences en compétences et connaissances peuvent changer.
- Le ton doit être factuel, lisible et raisonnable. Il ne s'agit pas de publicité.

Forme:
- 3 paragraphes, chacun avec 2 à 4 phrases.
- Pas de listes à puces.
- Pas de texte en dehors de ces trois paragraphes.

Évite explicitement:
- les formulations exagérées,
- les phrases marketing,
- les expressions suivantes: {PROHIBITED}

Donne uniquement les trois paragraphes, séparés par une ligne vide.
"""
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un auteur francophone spécialisé dans l'analyse de domaines techniques et thématiques, "
                "avec un style sobre et structuré."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw = call_deepseek(CONFIG, messages, max_tokens=CONFIG.max_tokens, temperature=0.7)
    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(parts) > 3:
        parts = parts[:3]
    return parts


# =========================
# Content helpers
# =========================

def _extract_area_from_heading(heading: str) -> str:
    """Extract area label from heading"""
    m = re.search(r"domaine\s+(.+)$", heading, flags=re.IGNORECASE)
    if not m:
        return heading
    return m.group(1).strip()


def has_ai_h2_marker(content: str) -> bool:
    """Check if a previous cross topic H2 marker already exists (simple substring check)"""
    return "Sous thème pour le domaine" in content


def build_insert_block(h2_text: str, paragraphs: List[str]) -> str:
    """Build insert block"""
    paragraphs_html = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        f'<h2>{h2_text}</h2>\n\n'
        f'<p><em>Remarque: Les appréciations ci dessous reposent sur une analyse d\'expert et décrivent '
        f'la situation actuelle telle qu\'elle peut être évaluée à partir des informations disponibles.</em></p>\n\n'
        f"{paragraphs_html}\n"
    )


def insert_at_end(content: str, insert_block: str) -> str:
    """Insert block at the end of the post"""
    return content.rstrip() + "\n\n" + insert_block.strip() + "\n"


# =========================
# Broken image check and prompt
# =========================

def find_broken_images_in_post(wp: WordPressClient, post: Dict[str, Any]) -> List[str]:
    """Check featured and content images and return unreachable URLs"""
    broken = []
    
    featured_id = post.get("featured_media") or 0
    if featured_id:
        media = wp.get_media(featured_id)
        if media:
            src = media.get("source_url")
            if src:
                try:
                    resp = requests.head(src, timeout=5)
                    if resp.status_code >= 400:
                        broken.append(src)
                except Exception:
                    broken.append(src)
    
    content = post.get("content", {}).get("rendered", "")
    img_urls = re.findall(r'<img[^>]+src="([^"]+)"', content)
    
    for url in img_urls:
        try:
            resp = requests.head(url, timeout=5)
            if resp.status_code >= 400:
                broken.append(url)
        except Exception:
            broken.append(url)
    
    return broken


def prompt_fix_broken_images(post_id: int, edit_link: str, post_link: str, broken_urls: List[str]) -> str:
    """
Show warning about broken images and wait for user confirmation.

Returns:
- "continue" to continue
- "skip" to skip this post
- "quit" to abort
"""
    print(f"\n{RED}{'=' * 60}{RESET}")
    print(f"{RED}⚠️  WARNING: BROKEN IMAGES IN POST {post_id}{RESET}")
    print(f"{RED}{'=' * 60}{RESET}")
    print(f"{BLUE}📝 Edit link: {edit_link}{RESET}")
    print(f"{BLUE}🔗 Frontend:  {post_link}{RESET}")
    print(f"{RED}{'─' * 60}{RESET}")
    
    for i, broken in enumerate(broken_urls, start=1):
        print(f"{RED}[{i}] {broken}{RESET}")
    
    print(f"{RED}{'=' * 60}{RESET}")
    
    if CONFIG.auto_all:
        print(f"{YELLOW}[AUTO]{RESET} Automatic mode, skipping confirmation")
        return "continue"
    
    while True:
        msg = (
            f"\n{YELLOW}Please fix the images (click the edit link above), then confirm.{RESET}\n"
            "[Enter]=Images fixed, continue | [s]=Skip post | [q]=Abort: "
        )
        choice = input(msg).strip().lower()
        
        if choice == "" or choice == "c":
            print(f"{GREEN}[OK]{RESET} Fix confirmed, continue\n")
            return "continue"
        elif choice == "s":
            print(f"{YELLOW}[SKIP]{RESET} Post {post_id} skipped because of broken images")
            return "skip"
        elif choice == "q":
            return "quit"


# =========================
# Skiplist and processed data
# =========================

def load_skip_data(wp: WordPressClient, config: Config) -> Tuple[Set[str], Set[str], Set[int]]:
    """Load skiplist URLs and processed IDs"""
    skiplist_urls: Set[str] = set()
    skip_nein: Set[str] = set()
    
    if config.skiplist_url_1:
        skiplist_urls = wp.load_skiplist_urls(config.skiplist_url_1)
    if config.skiplist_url_2:
        skip_nein = set(wp.load_skiplist(config.skiplist_url_2))
    
    ppm = ProcessedPostsManager(config.processed_log_path)
    processed = ppm.load()
    
    if skiplist_urls:
        print(f"{GREEN}[SKIP-URL1]{RESET} {len(skiplist_urls)} URLs in skiplist 1")
    if skip_nein:
        print(f"{GREEN}[SKIP-URL2]{RESET} {len(skip_nein)} patterns in skiplist 2")
    print(f"{GREEN}[PROCESSED]{RESET} {len(processed)} posts already processed")
    
    return skiplist_urls, skip_nein, processed


# =========================
# Main processing
# =========================

def process_post(
    wp: WordPressClient,
    post: Dict[str, Any],
    skiplist_urls: Set[str],
    skip_nein: Set[str],
    processed_mgr: ProcessedPostsManager,
) -> bool:
    """
Process a single post.

Returns True if the post was updated.
"""
    post_id = post["id"]
    link = post.get("link", "")
    title = post.get("title", {}).get("rendered", "")
    content = post.get("content", {}).get("rendered", "")
    
    if normalize_url(link) in skiplist_urls:
        print(f"{YELLOW}[SKIP]{RESET} {post_id} URL is in skiplist: {link}")
        return False
    
    for nein in skip_nein:
        if nein and nein in content:
            print(f"{YELLOW}[SKIP]{RESET} {post_id} local skip pattern")
            return False
    
    if not title or not content:
        print(f"{YELLOW}[SKIP]{RESET} {post_id} missing title or content")
        return False
    
    if len(strip_html_to_text(content)) < CONFIG.min_content_length:
        print(f"{YELLOW}[SKIP]{RESET} {post_id} content too short")
        return False
    
    blocked = [
        "AI-Impact Inserter",
        "AI-Impact Test",
        "Test AI-Impact",
    ]
    if any(b in title for b in blocked):
        print(f"{YELLOW}[SKIP]{RESET} {post_id} helper or test content")
        return False
    
    if CONFIG.only_title_contains and CONFIG.only_title_contains.lower() not in strip_html_to_text(title).lower():
        print(f"{YELLOW}[SKIP]{RESET} {post_id} title does not contain filter")
        return False
    
    if not is_relevant_post(title, content):
        print(f"{YELLOW}[SKIP]{RESET} {post_id} not relevant according to filter")
        return False
    
    if has_ai_h2_marker(content):
        print(f"{YELLOW}[SKIP]{RESET} {post_id} cross topic H2 heading already present")
        return False
    
    if has_ai_coverage(title, content):
        print(f"{YELLOW}[SKIP]{RESET} {post_id} cross topic already covered in detail")
        return False
    
    heading = generate_area_heading(title, content)
    heading = normalize_heading_with_deepseek(heading, title, content)
    
    area_label = _extract_area_from_heading(heading)
    sender_role = choose_sender_designation(title, content)
    opening_question = generate_opening_question(area_label)
    paragraphs = generate_ai_impact_paragraphs(title, content, sender_role, area_label)
    
    insert_block = (
        f"<h2>{heading}</h2>\n\n"
        f"<p><strong>{opening_question}</strong></p>\n\n"
        f"<p>{paragraphs[0]}</p>\n\n"
        f"<p>{paragraphs[1]}</p>\n\n"
        f"<p>{paragraphs[2]}</p>\n"
    )
    
    print(f"{BLUE}[H2]{RESET} {heading}")
    print(f"{BLUE}[LEN]{RESET} {len(insert_block)} characters")
    
    new_content = insert_at_end(content, insert_block)
    
    if os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes", "y"):
        print(f"{GREEN}[DRY]{RESET} {post_id} would update: {link}")
        return False
    
    try:
        updated = wp.update_post(post_id, {"content": new_content})
        final_url = updated.get("link", link)
        print(f"{GREEN}[OK]{RESET} {post_id} Updated: {final_url}")
        print(f"{GREEN}✅ {final_url}{RESET}\n")
        processed_mgr.mark(post_id)
        return True
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        print(f"{RED}[ERR]{RESET} {post_id} HTTP error: {e} | {body}")
    except Exception as e:
        print(f"{RED}[ERR]{RESET} {post_id} update failed: {e}")
    
    return False


def prompt_continue(updated_since_pause: int, total_updated: int) -> bool:
    """Ask user whether to continue after a batch of updated posts"""
    if CONFIG.auto_all:
        print(f"{GREEN}[MODE]{RESET} all posts without pause\n")
        return True
    
    msg = (
        f"\n{GREEN}[PAUSE]{RESET} {updated_since_pause} posts updated (total: {total_updated})\n"
        "Update 10 more? [Enter]=yes, [a]=all without pause, [q]=quit: "
    )
    choice = input(msg).strip().lower()
    
    if choice == "a":
        os.environ["AUTO_ALL"] = "1"
        CONFIG.auto_all = True
        print(f"{GREEN}[MODE]{RESET} all posts without pause\n")
        return True
    if choice == "q" or CONFIG.auto_quit:
        return False
    return True


# =========================
# Main loop
# =========================

def main():
    """Main entry point"""
    wp = WordPressClient(CONFIG)
    user_data = wp.get_user_info()
    
    active_base = wp.auth_mgr.active_base
    print(f"{GREEN}[BASE]{RESET} {active_base} | mode: {wp.auth_mgr.mode}\n")
    print(f"{GREEN}[USER]{RESET} {user_data.get('name')} | Roles: {user_data.get('roles')}\n")
    
    skiplist_urls, skip_nein, processed = load_skip_data(wp, CONFIG)
    
    print("═" * 60)
    print("WP AI-Impact Inserter v14.6")
    print("═" * 60)
    if CONFIG.auto_all:
        print(f"{GREEN}[MODE]{RESET} all posts without pause\n")
    print(f"{BLUE}[SORT]{RESET} processing posts by ascending post ID (smallest first)\n")
    print("─" * 60)
    
    page = 1
    total_processed = 0
    total_updated = 0
    stop = False
    
    while not stop:
        try:
            batch = wp.get_posts(page, CONFIG.per_page, CONFIG.statuses)
        except Exception as e:
            print(f"{RED}[ERR]{RESET} loading page {page}: {e}")
            break
        
        if not batch:
            print(f"{BLUE}[INFO]{RESET} no further posts")
            break
        
        print(f"\n{BLUE}[PAGE {page}]{RESET} {len(batch)} Posts")
        print("─" * 60)
        
        for post in batch:
            pid = post["id"]
            link = post.get("link", "")
            
            if pid in processed:
                print(f"{YELLOW}[SKIP]{RESET} {pid} already processed: {link}")
                continue
            
            broken = find_broken_images_in_post(wp, post)
            if broken:
                print(f"{BLUE}[CHECK]{RESET} {pid} checking images...")
                edit_link = f"{CONFIG.wp_base}/wp-admin/post.php?post={pid}&action=edit"
                post_link = link or f"{CONFIG.wp_base}/?p={pid}"
                
                action = prompt_fix_broken_images(pid, edit_link, post_link, broken)
                if action == "skip":
                    continue
                elif action == "quit":
                    stop = True
                    break
                else:
                    print(f"{GREEN}[IMG-OK]{RESET} {pid} all images reachable")
            else:
                print(f"{GREEN}[IMG-OK]{RESET} {pid} all images reachable")
            
            updated = process_post(
                wp,
                post,
                skiplist_urls,
                skip_nein,
                processed_mgr=ProcessedPostsManager(CONFIG.processed_log_path),
            )
            if updated:
                total_updated += 1
            
            total_processed += 1
            time.sleep(CONFIG.sleep_between)
        
        if stop:
            break
        
        page += 1
    
    print("\n" + "═" * 60)
    print("Run finished")
    print("═" * 60)
    print(f"Processed: {total_processed}")
    print(f"Updated: {total_updated}")
    print("═" * 60)


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[STOP]{RESET} stopped by user")
        sys.exit(130)
