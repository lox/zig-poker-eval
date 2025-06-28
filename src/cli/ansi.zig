const std = @import("std");

// ANSI escape codes
pub const ESC = "\x1b[";
pub const RESET = ESC ++ "0m";
pub const BOLD = ESC ++ "1m";
pub const DIM = ESC ++ "2m";
pub const ITALIC = ESC ++ "3m";
pub const UNDERLINE = ESC ++ "4m";

// Colors
pub const BLACK = ESC ++ "30m";
pub const RED = ESC ++ "31m";
pub const GREEN = ESC ++ "32m";
pub const YELLOW = ESC ++ "33m";
pub const BLUE = ESC ++ "34m";
pub const MAGENTA = ESC ++ "35m";
pub const CYAN = ESC ++ "36m";
pub const WHITE = ESC ++ "37m";

// Bright colors
pub const BRIGHT_BLACK = ESC ++ "90m";
pub const BRIGHT_RED = ESC ++ "91m";
pub const BRIGHT_GREEN = ESC ++ "92m";
pub const BRIGHT_YELLOW = ESC ++ "93m";
pub const BRIGHT_BLUE = ESC ++ "94m";
pub const BRIGHT_MAGENTA = ESC ++ "95m";
pub const BRIGHT_CYAN = ESC ++ "96m";
pub const BRIGHT_WHITE = ESC ++ "97m";

// Helper functions for common formatting
pub fn bold(text: []const u8) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}{s}{s}", .{ BOLD, text, RESET });
}

pub fn green(text: []const u8) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}{s}{s}", .{ GREEN, text, RESET });
}

pub fn red(text: []const u8) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}{s}{s}", .{ RED, text, RESET });
}

pub fn cyan(text: []const u8) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}{s}{s}", .{ CYAN, text, RESET });
}

pub fn yellow(text: []const u8) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}{s}{s}", .{ YELLOW, text, RESET });
}

pub fn blue(text: []const u8) [*:0]const u8 {
    return std.fmt.comptimePrint("{s}{s}{s}", .{ BLUE, text, RESET });
}

// Check if stdout supports colors
pub fn supportsColor() bool {
    if (std.posix.isatty(std.posix.STDOUT_FILENO)) {
        // Check TERM environment variable
        if (std.posix.getenv("TERM")) |term| {
            if (std.mem.indexOf(u8, term, "color") != null or
                std.mem.eql(u8, term, "xterm") or
                std.mem.eql(u8, term, "xterm-256color") or
                std.mem.eql(u8, term, "screen") or
                std.mem.eql(u8, term, "tmux"))
            {
                return true;
            }
        }
        // Default to true for TTY if no TERM info
        return true;
    }
    return false;
}

// Runtime formatting functions that respect color support
pub fn printBold(comptime fmt: []const u8, args: anytype) void {
    if (supportsColor()) {
        std.debug.print(BOLD ++ fmt ++ RESET, args);
    } else {
        std.debug.print(fmt, args);
    }
}

pub fn printGreen(comptime fmt: []const u8, args: anytype) void {
    if (supportsColor()) {
        std.debug.print(GREEN ++ fmt ++ RESET, args);
    } else {
        std.debug.print(fmt, args);
    }
}

pub fn printRed(comptime fmt: []const u8, args: anytype) void {
    if (supportsColor()) {
        std.debug.print(RED ++ fmt ++ RESET, args);
    } else {
        std.debug.print(fmt, args);
    }
}

pub fn printCyan(comptime fmt: []const u8, args: anytype) void {
    if (supportsColor()) {
        std.debug.print(CYAN ++ fmt ++ RESET, args);
    } else {
        std.debug.print(fmt, args);
    }
}

pub fn printYellow(comptime fmt: []const u8, args: anytype) void {
    if (supportsColor()) {
        std.debug.print(YELLOW ++ fmt ++ RESET, args);
    } else {
        std.debug.print(fmt, args);
    }
}

pub fn printBlue(comptime fmt: []const u8, args: anytype) void {
    if (supportsColor()) {
        std.debug.print(BLUE ++ fmt ++ RESET, args);
    } else {
        std.debug.print(fmt, args);
    }
}
