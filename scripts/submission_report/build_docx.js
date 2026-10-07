// 보고서 초안(markdown)을 Word 파일로 변환한다. 대회 양식 서식: 본문 휴먼명조 14pt / 줄간격 160%, 표·캡션 10pt.
//   node scripts/submission_report/build_docx.js [입력.md] [출력.docx]
// 지원 문법: #~### 제목, 문단, 표(|), 이미지 ![](경로), 인용(>), 목록(- , 1. ), 들여쓴 코드, **굵게**, `코드`.
// 블라인드 규칙: 문서 속성의 작성자·회사·설명은 비워 둔다.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  ImageRun, AlignmentType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const input = process.argv[2] || path.join(ROOT, "report/submission_draft/report_draft_v1.md");
const output = process.argv[3] || path.join(ROOT, "submission/report/report_draft_v1.docx");
const baseDir = path.dirname(input);

const FONT = "휴먼명조";
const BODY = 28;       // 14pt (half-points)
const SMALL = 20;      // 10pt
const LINE = 384;      // 160%
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134;   // A4, 2cm 여백
const TEXT_W = PAGE_W - 2 * MARGIN;

function runs(text, size, extra = {}) {
  // **굵게**와 `코드`를 나눠 TextRun 배열로 만든다.
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), size, font: FONT, ...extra }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, size, font: FONT, ...extra }));
    else out.push(new TextRun({ text: tok.slice(1, -1), size, font: "Consolas", ...extra }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), size, font: FONT, ...extra }));
  return out;
}

function para(text, opts = {}) {
  const size = opts.size || BODY;
  return new Paragraph({
    children: runs(text, size, opts.run || {}),
    spacing: { line: opts.line || LINE, after: opts.after ?? 120 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.indent,
    numbering: opts.numbering,
    shading: opts.shading,
    border: opts.border,
    keepNext: opts.keepNext,
  });
}

function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), buf };
}

function image(rel) {
  const file = path.resolve(baseDir, rel);
  if (!fs.existsSync(file)) return para(`〔그림 파일 없음: ${rel}〕`, { size: SMALL });
  const { w, h, buf } = pngSize(file);
  const maxW = 620;   // px at 96dpi ≈ 16.4cm
  const width = Math.min(maxW, w);
  const height = Math.round(h * width / w);
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: buf, transformation: { width, height } })],
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
  });
}

function table(lines) {
  const parse = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  const header = parse(lines[0]);
  const rows = lines.slice(2).map(parse);
  const n = header.length;
  // 열 너비: 내용 길이에 비례(최소 비중 보장), 합계 = 본문 폭
  const weight = Array.from({ length: n }, (_, i) => {
    const lens = [header, ...rows].map((r) => Array.from((r[i] || "").replace(/\*\*/g, "")).length);
    return Math.max(7, Math.min(40, Math.max(...lens)));
  });
  const total = weight.reduce((a, b) => a + b, 0);
  const widths = weight.map((wgt) => Math.floor(TEXT_W * wgt / total));
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const align = parse(lines[1]).map((c) => (c.endsWith(":") && !c.startsWith(":") ? AlignmentType.RIGHT : AlignmentType.LEFT));
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const cell = (text, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "E7EDF3", color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    borders: { top: border, bottom: border, left: border, right: border },
    children: [new Paragraph({ children: runs(text, SMALL, head ? { bold: true } : {}),
      alignment: head ? AlignmentType.CENTER : align[i], spacing: { line: 276, after: 0 } })],
  });
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
      ...rows.map((r) => new TableRow({ children: header.map((_, i) => cell(r[i] || "", i, false)) }))],
  });
}

function convert(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0, chapter = 0, listId = 0, prevNumbered = false;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^#\s/.test(line)) {
      const text = line.replace(/^#\s+/, "");
      if (chapter > 0) out.push(new Paragraph({ children: [new PageBreak()] }));
      chapter++;
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: chapter === 1 ? text : `□ ${text}`, font: FONT, size: 32, bold: true })],
        spacing: { before: 120, after: 240 } }));
      i++; continue;
    }
    if (/^##\s/.test(line)) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: line.replace(/^##\s+/, ""), font: FONT, size: 30, bold: true })],
        spacing: { before: 240, after: 120 }, keepNext: true }));
      i++; continue;
    }
    if (/^###\s/.test(line)) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: line.replace(/^###\s+/, ""), font: FONT, size: 28, bold: true })],
        spacing: { before: 200, after: 100 }, keepNext: true }));
      i++; continue;
    }
    if (/^\|/.test(line)) {
      const block = [];
      while (i < lines.length && /^\|/.test(lines[i])) block.push(lines[i++]);
      out.push(table(block));
      out.push(new Paragraph({ children: [], spacing: { after: 120 } }));
      continue;
    }
    const img = line.match(/^!\[[^\]]*\]\(([^)]+)\)/);
    if (img) { out.push(image(img[1])); i++; continue; }
    if (/^>\s?/.test(line)) {
      const block = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) block.push(lines[i++].replace(/^>\s?/, ""));
      out.push(para(block.join(" "), { size: 24, shading: { type: ShadingType.CLEAR, fill: "F1F4F8", color: "auto" },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: "1F6FB2", space: 6 } }, indent: { left: 200 } }));
      continue;
    }
    if (/^ {4}\S/.test(line)) {
      while (i < lines.length && /^ {4}/.test(lines[i])) {
        out.push(new Paragraph({ children: [new TextRun({ text: lines[i].slice(4), font: "Consolas", size: SMALL })],
          spacing: { line: 276, after: 0 }, shading: { type: ShadingType.CLEAR, fill: "F5F5F5", color: "auto" } }));
        i++;
      }
      out.push(new Paragraph({ children: [], spacing: { after: 120 } }));
      continue;
    }
    if (/^- /.test(line)) {
      out.push(para(line.replace(/^- /, ""), { numbering: { reference: "bullets", level: 0 }, after: 60 }));
      i++; continue;
    }
    if (/^\d+\.\s/.test(line)) {
      if (!prevNumbered) listId++;
      out.push(para(line.replace(/^\d+\.\s/, ""), { numbering: { reference: `num${listId}`, level: 0 }, after: 60 }));
      prevNumbered = /^\d+\.\s/.test(lines[i + 1] || "");
      i++; continue;
    }
    if (/^그림 \d|^표 \d/.test(line)) {
      out.push(para(line, { size: SMALL, align: AlignmentType.CENTER, after: 200 }));
      i++; continue;
    }
    if (/^ {2}\S/.test(line)) {   // 수식 줄
      out.push(para(line.trim(), { align: AlignmentType.CENTER }));
      i++; continue;
    }
    out.push(para(line));
    i++;
  }
  return { children: out, lists: listId };
}

const md = fs.readFileSync(input, "utf-8");
const { children, lists } = convert(md);
const numConfigs = [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "-", alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] }];
for (let c = 1; c <= lists; c++) {
  numConfigs.push({ reference: `num${c}`, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 440, hanging: 300 } } } }] });
}

const doc = new Document({
  creator: "", lastModifiedBy: "", title: "결과 보고서 초안", description: "", subject: "", keywords: "",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  numbering: { config: numConfigs },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: SMALL, font: FONT })] })] }) },
    children,
  }],
});

fs.mkdirSync(path.dirname(output), { recursive: true });
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(output, buf); console.log(`작성: ${path.relative(ROOT, output)} (${buf.length} bytes)`); });
