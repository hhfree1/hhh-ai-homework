"""把 posts/*.md 构建成静态博客。

三步加一篇文章（页面上的「取舍」页也这么写）：

    1. 写 posts/<slug>.md
    2. 在 posts/posts.json 里加一条元数据
    3. python build_blog.py

产物：index.html、post/<slug>.html、about.html、differences.html、feed.xml。
没有时间戳，没有随机数：同样的输入连跑两次，产物逐字节一致。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POSTS_DIR = ROOT / "posts"
TPL = ROOT / "templates"

# 部署时只改这一行：RSS 的绝对地址（没有它，阅读器里的链接是死的）
BASE_URL = "https://hhfree1.github.io/hhh-ai-homework/hhh-blog/"

TZ = timezone(timedelta(hours=8))
WEEKDAY = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# ---------------------------------------------------------------- markdown

def inline(text: str) -> str:
    s = html.escape(text, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    return s


def render_md(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + inline("".join(para)) + "</p>")
            para.clear()

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("```"):
            flush()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf), quote=False) + "</code></pre>")
            continue
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("### "):
            flush()
            out.append("<h3>" + inline(s[4:]) + "</h3>")
            i += 1
            continue
        if s.startswith("## "):
            flush()
            out.append("<h2>" + inline(s[3:]) + "</h2>")
            i += 1
            continue
        if s in ("---", "***"):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if s.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append("<li>" + inline(lines[i].strip()[2:]) + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\d+\.\s", s):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append("<li>" + inline(re.sub(r"^\d+\.\s", "", lines[i].strip())) + "</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        if s.startswith("> "):
            flush()
            buf = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                buf.append(lines[i].strip()[2:])
                i += 1
            out.append("<blockquote><p>" + inline("".join(buf)) + "</p></blockquote>")
            continue
        para.append(s)
        i += 1

    flush()
    return "\n".join(out)


def wordcount(body_html: str) -> int:
    text = re.sub(r"<[^>]+>", "", body_html)
    text = html.unescape(text)
    return len(re.sub(r"\s", "", text))


# ---------------------------------------------------------------- 渲染

def fill(tpl: str, values: dict[str, str]) -> str:
    out = tpl
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", out)
    if left:
        raise SystemExit(f"模板里还有没填的占位符：{sorted(set(left))}")
    return out


def date_rfc822(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=TZ)
    return d.strftime("%a, %d %b %Y %H:%M:%S +0800")


def date_label(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d")
    return f"{iso} {WEEKDAY[d.weekday()]}"


def build() -> None:
    meta_all = json.loads((POSTS_DIR / "posts.json").read_text(encoding="utf-8"))
    site = meta_all["site"]
    posts = meta_all["posts"]
    posts.sort(key=lambda p: p["date"], reverse=True)

    count = len(posts)
    updated = posts[0]["date"]
    author = site.get("author", "hhfree1")

    head_tpl = (TPL / "head.html").read_text(encoding="utf-8")
    foot_tpl = (TPL / "foot.html").read_text(encoding="utf-8")

    def page(title: str, desc: str, root: str, body: str) -> str:
        head = fill(head_tpl, {"TITLE": title, "DESC": desc, "ROOT": root})
        foot = fill(foot_tpl, {"ROOT": root, "COUNT": str(count), "UPDATED": updated, "AUTHOR": author})
        return head + body + foot

    # ---- 首页
    rows = []
    for p in posts:
        body_md = (POSTS_DIR / f"{p['slug']}.md").read_text(encoding="utf-8")
        wc = wordcount(render_md(body_md))
        tags = " · ".join(p["tags"])
        rows.append(
            '    <li>\n'
            '      <article class="post-row reveal">\n'
            f'        <time class="post-row__date" datetime="{p["date"]}">{p["date"]}</time>\n'
            '        <div>\n'
            f'          <h2 class="post-row__title"><a href="post/{p["slug"]}.html">{html.escape(p["title"])}</a></h2>\n'
            f'          <p class="post-row__sum">{html.escape(p["summary"])}</p>\n'
            '        </div>\n'
            f'        <span class="post-row__meta">{wc} 字<br>{html.escape(tags)}</span>\n'
            '      </article>\n'
            '    </li>'
        )

    index_body = fill((TPL / "index.html").read_text(encoding="utf-8"), {
        "ROWS": "\n".join(rows),
        "COUNT": str(count),
        "UPDATED": updated,
    })
    (ROOT / "index.html").write_text(
        page(f"{site['title']} · 全部文章", site["desc"], "", index_body), encoding="utf-8")

    # ---- 文章页
    post_tpl = (TPL / "post.html").read_text(encoding="utf-8")
    for i, p in enumerate(posts):
        md = (POSTS_DIR / f"{p['slug']}.md").read_text(encoding="utf-8")
        body_html = render_md(md)
        pager = []
        if i > 0:
            n = posts[i - 1]
            pager.append(
                f'      <a class="pager__item" href="../post/{n["slug"]}.html">'
                f'<span class="pager__k">较新一篇</span><span class="pager__t">{html.escape(n["title"])}</span></a>')
        else:
            pager.append('      <span></span>')
        if i + 1 < len(posts):
            o = posts[i + 1]
            pager.append(
                f'      <a class="pager__item pager__item--next" href="../post/{o["slug"]}.html">'
                f'<span class="pager__k">较早一篇</span><span class="pager__t">{html.escape(o["title"])}</span></a>')
        else:
            pager.append('      <span></span>')

        body = fill(post_tpl, {
            "DATE_LABEL": date_label(p["date"]),
            "TAGS": html.escape(" · ".join(p["tags"])),
            "TITLE": html.escape(p["title"]),
            "SUMMARY": html.escape(p["summary"]),
            "WC": str(wordcount(body_html)),
            "SLUG": p["slug"],
            "BODY": body_html,
            "PAGER": "\n".join(pager),
        })
        out = page(f"{p['title']} · {site['title']}", p["summary"], "../", body)
        (ROOT / "post" / f"{p['slug']}.html").write_text(out, encoding="utf-8")

    # ---- 关于 / 取舍
    for name, title in (("about", "关于"), ("differences", "取舍")):
        body = fill((TPL / f"{name}.html").read_text(encoding="utf-8"), {
            "ROOT": "", "COUNT": str(count), "UPDATED": updated, "AUTHOR": author,
        })
        (ROOT / f"{name}.html").write_text(
            page(f"{title} · {site['title']}", site["desc"], "", body), encoding="utf-8")

    # ---- RSS
    items = []
    for p in posts:
        link = f"{BASE_URL}post/{p['slug']}.html"
        items.append(
            "    <item>\n"
            f"      <title>{html.escape(p['title'])}</title>\n"
            f"      <link>{link}</link>\n"
            f'      <guid isPermaLink="true">{link}</guid>\n'
            f"      <pubDate>{date_rfc822(p['date'])}</pubDate>\n"
            f"      <dc:creator>{html.escape(author)}</dc:creator>\n"
            f"      <description>{html.escape(p['summary'])}</description>\n"
            "    </item>"
        )
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        "  <channel>\n"
        f"    <title>{html.escape(site['title'])}</title>\n"
        f"    <link>{BASE_URL}</link>\n"
        f"    <description>{html.escape(site['desc'])}</description>\n"
        "    <language>zh-CN</language>\n"
        f"    <managingEditor>{html.escape(author)}</managingEditor>\n"
        f"    <lastBuildDate>{date_rfc822(updated)}</lastBuildDate>\n"
        f'    <atom:link href="{BASE_URL}feed.xml" rel="self" type="application/rss+xml"/>\n'
        + "\n".join(items) + "\n"
        "  </channel>\n"
        "</rss>\n"
    )
    (ROOT / "feed.xml").write_text(feed, encoding="utf-8")

    # ---- 指纹：posts.json + 每篇 md 的 sha256，改内容一定变
    h = hashlib.sha256()
    h.update((POSTS_DIR / "posts.json").read_bytes())
    for p in posts:
        h.update((POSTS_DIR / f"{p['slug']}.md").read_bytes())
    print(f"构建完成 · {count} 篇 · 最近更新 {updated} · 内容指纹 {h.hexdigest()[:12]}")


def main() -> int:
    try:
        build()
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
