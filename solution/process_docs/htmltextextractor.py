from dataclasses import dataclass
import re
from bs4 import BeautifulSoup, Tag
import trafilatura
from markdown_it import MarkdownIt


@dataclass
class MdHeadingInfo:
    level: int
    text: str
    start: int


class HtmlTextExtractor:
    # Служит для извлечения содержательного текста из HTML разметки
    # 1. С помощью BeautifulSoup из HTML удаляются ненужные тэги
    # 2. С помощью trafilatura, содержимое конвертируется в markdown
    # 3. С помощью markdown-it-py удаляются лишние разделы
    # 4. Удаляются стили и лишние символы (типа \n)
    def extract_markdown_from_html(self, html_content: str):
        soup = BeautifulSoup(html_content, "lxml")
        main = soup.find("main")
        header_title = ""
        if main:
            # Определяем заголовок H1, trafilatura почему-то пропускает его
            header_title = self._extract_header_title(main)
            # Удаляем сноски, цитаты изображения и другие ненужные тэги
            self._remove_unnecessary_tags(main)
            # Оставляем только секцию main
            html_content = f"<!DOCTYPE html><html lang=\"en\"><body>{str(main)}</body></html>"

        # Извлекаем содержательный текст в виде markdown
        md = trafilatura.extract(html_content, output_format="markdown", include_links=False, include_formatting=True)
        if md is None:
            md = ""

        # trafilatura почему-то пропускает H1, добавляем его вручную
        if header_title:
            md = f"# {header_title}\n\n{md}"

        # Удаляем ненужные разделы
        for title in ["Behind the scenes", "Appearances", "Sources", "Notes and references", "External links"]:
            md = self._remove_markdown_section(md, title)

        # Удаляем маркеры стиля
        md = self._remove_markdown_style_markers(md)

        # Удаляем лишние символы
        md = self._remove_unnecessary_symbols(md)

        return md

    # Извлекает заголовок
    def _extract_header_title(self, main: Tag):
        h1 = main.find('h1', class_='page-header__title', id='firstHeading')
        if not h1:
            return ""
        for span in h1.find_all("span"):
            span.decompose()
        return h1.get_text(strip=True)

    # Удаляет лишние тэги с ненужной информацией
    def _remove_unnecessary_tags(self, main: Tag):
        for tag in main.find_all(["img", "picture", "source", "svg", "video", "audio", "iframe", "script", "style", "noscript"]):
            tag.decompose()
        for quote in main.find_all("div", class_="quote"):
            quote.decompose()
        for span in main.find_all("span", class_="mw-editsection"):
            span.decompose()
        for sup in main.find_all("sup", class_="reference"):
            sup.decompose()

    # Удаляет раздел
    def _remove_markdown_section(self, md: str, title: str):
        lines = md.splitlines(keepends=True)
        tokens = MarkdownIt().parse(md)

        headings: list[MdHeadingInfo] = []
        for i, tok in enumerate(tokens):
            if tok.type == "heading_open":
                headings.append(MdHeadingInfo(level=int(tok.tag[1]),
                                text=tokens[i + 1].content.strip(), start=tok.map[0]))

        start = None
        end = None
        for cur in headings:
            if cur.text == title:
                start = cur.start
                end = len(lines)
                for other in headings:
                    if other.start > start and other.level <= cur.level:
                        end = other.start
                        break
                break

        if start is None or end is None:
            return md

        return "".join(lines[:start] + lines[end:])

    # Удаляет маркеры стиля
    def _remove_markdown_style_markers(self, md: str):
        md = re.sub(r"~~(.+?)~~", r"\1", md, flags=re.DOTALL)                               # ~~xxx~~
        md = re.sub(r"\*\*(.+?)\*\*", r"\1", md, flags=re.DOTALL)                           # **xxx**
        md = re.sub(r"__(.+?)__", r"\1", md, flags=re.DOTALL)                               # __xxx__
        md = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", md, flags=re.DOTALL)     # *xxx*
        md = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", md, flags=re.DOTALL)           # _xxx_
        md = re.sub(r"`(.+?)`", r"\1", md, flags=re.DOTALL)                                 # `xxx`
        return md

    # Удаляет лишние символы
    def _remove_unnecessary_symbols(self, md: str):
        md = re.sub(r"(?:\\n)+", "", md)  # это не перенос строки это именно слэш+n в тексте
        return md
