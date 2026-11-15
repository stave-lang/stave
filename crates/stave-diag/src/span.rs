//! Source spans, the file table, and the line index used to print `line:col`.

/// Index of a file in a [`SourceFiles`] table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

/// A byte range of one source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub file: FileId,
    pub offset: u32,
    pub len: u32,
}

impl Span {
    pub fn new(file: FileId, offset: u32, len: u32) -> Span {
        Span { file, offset, len }
    }
}

/// Byte offsets of line starts, for converting an offset into a 1-based line and a 1-based column
/// counted in UTF-8 bytes. Lines end at `\n` only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineIndex {
    line_starts: Vec<u32>,
    len: u32,
}

impl LineIndex {
    /// Indexes `text`. A text longer than `u32::MAX` bytes (E0002) is indexed up to that size,
    /// so a span at its start still prints.
    pub fn new(text: &[u8]) -> LineIndex {
        let limit = text.len().min(u32::MAX as usize);
        let mut line_starts = vec![0u32];
        for (i, &b) in text[..limit].iter().enumerate() {
            if b == b'\n' {
                // i < limit <= u32::MAX, so i + 1 always fits.
                if let Ok(next) = u32::try_from(i + 1) {
                    line_starts.push(next);
                }
            }
        }
        LineIndex { line_starts, len: limit as u32 }
    }

    /// The indexed length in bytes.
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of lines (a text without newline has one).
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// 1-based `(line, col)` of `offset`; `offset == len` (end of file) is allowed.
    /// `None` when the offset lies past the end.
    pub fn line_col(&self, offset: u32) -> Option<(u32, u32)> {
        if offset > self.len {
            return None;
        }
        let line = self.line_starts.partition_point(|&s| s <= offset);
        // line >= 1 because line_starts[0] == 0 <= offset.
        let start = *self.line_starts.get(line.checked_sub(1)?)?;
        let line = u32::try_from(line).ok()?;
        Some((line, offset - start + 1))
    }
}

/// One registered source file.
#[derive(Clone, Debug)]
pub struct SourceFile {
    pub package: String,
    pub path: String,
    pub lines: LineIndex,
}

/// The files diagnostics point into: the package and path of each (the diagnostic sort key) and its
/// line index.
#[derive(Clone, Debug, Default)]
pub struct SourceFiles {
    files: Vec<SourceFile>,
}

/// Error from [`SourceFiles::add`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileTableError {
    /// More than `u32::MAX` files.
    TooManyFiles,
}

impl std::fmt::Display for FileTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileTableError::TooManyFiles => write!(f, "more than {} source files", u32::MAX),
        }
    }
}

impl std::error::Error for FileTableError {}

impl SourceFiles {
    pub fn new() -> SourceFiles {
        SourceFiles::default()
    }

    /// Registers a file and indexes its lines. `text` may be invalid UTF-8 (E0001).
    pub fn add(&mut self, package: &str, path: &str, text: &[u8]) -> Result<FileId, FileTableError> {
        let id = u32::try_from(self.files.len()).map_err(|_| FileTableError::TooManyFiles)?;
        self.files.push(SourceFile {
            package: package.to_string(),
            path: path.to_string(),
            lines: LineIndex::new(text),
        });
        Ok(FileId(id))
    }

    pub fn get(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

/// A span resolved for printing: file path, 1-based line and column (UTF-8 bytes), length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location<'a> {
    pub path: &'a str,
    pub line: u32,
    pub col: u32,
    pub len: u32,
}

/// A span that cannot be printed: its file is not in the table, or its offset lies past the end
/// of the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocateError {
    UnknownFile(FileId),
    OffsetOutOfRange { file: FileId, offset: u32, len: u32 },
}

impl std::fmt::Display for LocateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocateError::UnknownFile(id) => write!(f, "span in unknown file {}", id.0),
            LocateError::OffsetOutOfRange { file, offset, len } => {
                write!(f, "span offset {offset} past the end of file {} ({len} bytes)", file.0)
            }
        }
    }
}

impl std::error::Error for LocateError {}

impl SourceFiles {
    /// Resolves `span` to its path, line, and column.
    pub fn locate(&self, span: Span) -> Result<Location<'_>, LocateError> {
        let f = self.get(span.file).ok_or(LocateError::UnknownFile(span.file))?;
        let (line, col) = f.lines.line_col(span.offset).ok_or(LocateError::OffsetOutOfRange {
            file: span.file,
            offset: span.offset,
            len: f.lines.len(),
        })?;
        Ok(Location { path: &f.path, line, col, len: span.len })
    }
}
