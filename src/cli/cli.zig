const std = @import("std");

// Generic CLI framework using comptime reflection
pub fn Cli(comptime CommandsType: type) type {
    return struct {
        const Self = @This();

        // Extract command definitions from the type
        pub const Commands = CommandsType;
        pub const CommandTag = std.meta.Tag(Commands);

        // Result type for parsing
        pub const ParseResult = union(enum) {
            command: Commands,
            help_main,
            help_command: CommandTag,
        };

        // Parse command line arguments
        pub fn parseArgs(allocator: std.mem.Allocator, args: [][:0]u8) !ParseResult {
            if (args.len < 2) return ParseResult{ .help_main = {} };

            const command_str = args[1];

            // Handle global help flags
            if (std.mem.eql(u8, command_str, "--help") or std.mem.eql(u8, command_str, "-h") or std.mem.eql(u8, command_str, "help")) {
                return ParseResult{ .help_main = {} };
            }

            // Parse subcommand
            const tag = std.meta.stringToEnum(CommandTag, command_str) orelse {
                return error.InvalidCommand;
            };

            // Check for command-specific help
            for (args[2..]) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    return ParseResult{ .help_command = tag };
                }
            }

            // Parse the actual command using comptime dispatch
            const command = try parseCommand(tag, allocator, args[2..]);
            return ParseResult{ .command = command };
        }

        // Generic command parser using comptime reflection on union
        fn parseCommand(tag: CommandTag, allocator: std.mem.Allocator, args: [][:0]u8) !Commands {
            inline for (std.meta.fields(Commands)) |field| {
                if (@field(CommandTag, field.name) == tag) {
                    if (field.type == void) {
                        return @unionInit(Commands, field.name, {});
                    } else {
                        const parsed = try parseSubcommand(field.type, allocator, args);
                        return @unionInit(Commands, field.name, parsed);
                    }
                }
            }
            unreachable;
        }

        // Generic subcommand parser using comptime reflection
        fn parseSubcommand(comptime T: type, allocator: std.mem.Allocator, args: [][:0]u8) !T {
            var result: T = std.mem.zeroInit(T, .{});
            var positional_count: usize = 0;

            // Initialize allocator-dependent fields
            inline for (std.meta.fields(T)) |field| {
                if (comptime std.mem.indexOf(u8, @typeName(field.type), "ArrayList")) |_| {
                    @field(result, field.name) = @TypeOf(@field(result, field.name)).init(allocator);
                }
            }

            var i: usize = 0;
            while (i < args.len) {
                const arg = args[i];

                if (std.mem.startsWith(u8, arg, "--")) {
                    // Handle long options
                    const option_name = arg[2..];
                    var option_value: ?[]const u8 = null;

                    // Check for --option=value format
                    if (std.mem.indexOf(u8, option_name, "=")) |eq_pos| {
                        const name = option_name[0..eq_pos];
                        const value = option_name[eq_pos + 1 ..];
                        option_value = value;
                        try setFieldValue(T, &result, name, option_value);
                    } else {
                        // Check if next arg is the value
                        if (i + 1 < args.len and !std.mem.startsWith(u8, args[i + 1], "-")) {
                            option_value = args[i + 1];
                            i += 1;
                        }
                        try setFieldValue(T, &result, option_name, option_value);
                    }
                } else {
                    // Handle positional arguments
                    try setPositionalArg(T, &result, arg, &positional_count);
                }
                i += 1;
            }

            return result;
        }

        // Set field value using comptime reflection
        fn setFieldValue(comptime T: type, result: *T, option_name: []const u8, value: ?[]const u8) !void {
            inline for (std.meta.fields(T)) |field| {
                if (std.mem.eql(u8, field.name, option_name)) {
                    try setTypedFieldValue(T, result, field.name, field.type, value);
                    return;
                }
            }
            // Unknown option - could return error or ignore
        }

        // Set field value based on type
        fn setTypedFieldValue(comptime T: type, result: *T, comptime field_name: []const u8, comptime FieldType: type, value: ?[]const u8) !void {
            switch (@typeInfo(FieldType)) {
                .bool => {
                    @field(result, field_name) = true;
                },
                .optional => |opt| {
                    if (value) |v| {
                        switch (@typeInfo(opt.child)) {
                            .bool => @field(result, field_name) = true,
                            .int => @field(result, field_name) = try std.fmt.parseInt(opt.child, v, 10),
                            .pointer => @field(result, field_name) = v,
                            else => return error.UnsupportedOptionalType,
                        }
                    }
                },
                .int => {
                    if (value) |v| {
                        @field(result, field_name) = try std.fmt.parseInt(FieldType, v, 10);
                    } else {
                        return error.MissingValue;
                    }
                },
                .pointer => |ptr| {
                    if (ptr.child == u8) { // String type
                        if (value) |v| {
                            @field(result, field_name) = v;
                        } else {
                            return error.MissingValue;
                        }
                    }
                },
                .@"enum" => {
                    if (value) |v| {
                        @field(result, field_name) = std.meta.stringToEnum(FieldType, v) orelse return error.InvalidEnumValue;
                    } else {
                        return error.MissingValue;
                    }
                },
                else => return error.UnsupportedFieldType,
            }
        }

        // Set positional arguments - requires user to define getPositionalFields
        fn setPositionalArg(comptime T: type, result: *T, arg: []const u8, positional_count: *usize) !void {
            if (!@hasDecl(Commands, "getPositionalFields")) {
                return; // No positional args defined
            }

            const positional_fields = Commands.getPositionalFields(T);

            if (positional_count.* >= positional_fields.len) {
                return error.TooManyPositionalArgs;
            }

            const field_name = positional_fields[positional_count.*];
            positional_count.* += 1;

            inline for (std.meta.fields(T)) |field| {
                if (std.mem.eql(u8, field.name, field_name)) {
                    try setTypedFieldValue(T, result, field.name, field.type, arg);
                    return;
                }
            }
        }

        // Print main help (list of all commands)
        pub fn printMainHelp() void {
            if (!@hasDecl(Commands, "getAppMeta")) {
                @panic("Commands must define getAppMeta() function");
            }

            const app_meta = Commands.getAppMeta();
            const ansi = @import("ansi.zig");
            const print = std.debug.print;

            ansi.printBold("🃏 {s}\n", .{app_meta.name});
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
            print("{s}\n", .{app_meta.description});

            ansi.printBold("\n📖 Usage:\n", .{});
            print("  {s} <command> [options] [args...]\n", .{app_meta.name});
            print("  {s} <command> --help    # Show command-specific help\n", .{app_meta.name});

            ansi.printBold("\n⚙️  Commands:\n", .{});

            // Generate command list using comptime reflection
            inline for (std.meta.fields(Commands)) |field| {
                const tag = @field(CommandTag, field.name);
                const meta = Commands.getCommandMeta(tag);
                ansi.printCyan("  {s:<8}", .{meta.name});
                print("{s}\n", .{meta.description});
            }

            if (app_meta.global_options.len > 0) {
                ansi.printBold("\n🔧 Global Options:\n", .{});
                for (app_meta.global_options) |opt| {
                    ansi.printYellow("  {s}", .{opt.flag});
                    print("{s:<20} {s}\n", .{ "", opt.description });
                }
            }

            if (app_meta.examples.len > 0) {
                ansi.printBold("\n💡 Examples:\n", .{});
                for (app_meta.examples) |example| {
                    print("  {s}\n", .{example});
                }
            }
        }

        // Print command-specific help
        pub fn printCommandHelp(tag: CommandTag) void {
            if (!@hasDecl(Commands, "getCommandMeta")) {
                @panic("Commands must define getCommandMeta() function");
            }

            const ansi = @import("ansi.zig");
            const print = std.debug.print;
            const meta = Commands.getCommandMeta(tag);

            ansi.printBold("🃏 {s}\n", .{meta.name});
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
            print("{s}\n", .{meta.description});

            ansi.printBold("\n📖 Usage:\n", .{});
            print("  {s}\n", .{meta.usage});

            // Print options for commands that have option structs
            inline for (std.meta.fields(Commands)) |field| {
                if (@field(CommandTag, field.name) == tag and field.type != void) {
                    printOptionsHelp(field.type, tag);
                    break;
                }
            }

            // Print examples
            if (meta.examples.len > 0) {
                ansi.printBold("\n💡 Examples:\n", .{});
                for (meta.examples) |example| {
                    print("  {s}\n", .{example});
                }
            }
        }

        fn printOptionsHelp(comptime T: type, tag: CommandTag) void {
            const ansi = @import("ansi.zig");
            const print = std.debug.print;

            // Get positional fields at comptime
            const positional_fields = comptime if (@hasDecl(Commands, "getPositionalFields"))
                Commands.getPositionalFields(T)
            else
                &[_][]const u8{};

            // Print positional arguments first
            if (positional_fields.len > 0) {
                ansi.printBold("\n📝 Arguments:\n", .{});
                inline for (positional_fields) |field_name| {
                    inline for (std.meta.fields(T)) |field| {
                        if (comptime std.mem.eql(u8, field.name, field_name)) {
                            ansi.printGreen("  {s:<12}", .{field_name});
                            print("{s}\n", .{getFieldDescription(field.name)});
                        }
                    }
                }
            }

            _ = tag; // unused for now

            // Print optional flags/options
            ansi.printBold("\n🔧 Options:\n", .{});

            // Calculate max option width for alignment
            comptime var max_option_width: usize = 0;
            inline for (std.meta.fields(T)) |field| {
                const is_positional = comptime blk: {
                    for (positional_fields) |pos_field| {
                        if (std.mem.eql(u8, field.name, pos_field)) break :blk true;
                    }
                    break :blk false;
                };

                if (!is_positional) {
                    const takes_value = switch (@typeInfo(field.type)) {
                        .bool => false,
                        .optional => |opt| switch (@typeInfo(opt.child)) {
                            .bool => false,
                            else => true,
                        },
                        else => true,
                    };

                    // Calculate option string length: "--fieldname" + " <value>" if needed
                    const option_len = 2 + field.name.len + (if (takes_value) 8 else 0); // " <value>" = 8 chars
                    if (option_len > max_option_width) {
                        max_option_width = option_len;
                    }
                }
            }

            // Add some padding
            const total_width = max_option_width + 4;

            inline for (std.meta.fields(T)) |field| {
                // Skip positional arguments
                const is_positional = comptime blk: {
                    for (positional_fields) |pos_field| {
                        if (std.mem.eql(u8, field.name, pos_field)) break :blk true;
                    }
                    break :blk false;
                };

                if (!is_positional) {
                    const takes_value = switch (@typeInfo(field.type)) {
                        .bool => false,
                        .optional => |opt| switch (@typeInfo(opt.child)) {
                            .bool => false,
                            else => true,
                        },
                        else => true,
                    };

                    // Build the option string
                    var option_buf: [64]u8 = undefined;
                    const option_str = if (takes_value)
                        std.fmt.bufPrint(option_buf[0..], "--{s} <value>", .{field.name}) catch field.name
                    else
                        std.fmt.bufPrint(option_buf[0..], "--{s}", .{field.name}) catch field.name;

                    ansi.printYellow("  {s}", .{option_str});

                    // Calculate remaining spaces for alignment
                    const spaces_needed = if (option_str.len + 2 < total_width)
                        total_width - option_str.len - 2
                    else
                        2;

                    // Print spaces and description
                    var i: usize = 0;
                    while (i < spaces_needed) : (i += 1) {
                        print(" ", .{});
                    }
                    print("{s}\n", .{getFieldDescription(field.name)});
                }
            }
        }

        // Field descriptions - requires user to define getFieldDescription
        fn getFieldDescription(field_name: []const u8) []const u8 {
            if (@hasDecl(Commands, "getFieldDescription")) {
                return Commands.getFieldDescription(field_name);
            }
            return "No description available";
        }
    };
}

// Metadata types for defining app and command information
pub const AppMeta = struct {
    name: []const u8,
    description: []const u8,
    global_options: []const GlobalOption = &.{},
    examples: []const []const u8 = &.{},
};

pub const GlobalOption = struct {
    flag: []const u8,
    description: []const u8,
};

pub const CommandMeta = struct {
    name: []const u8,
    description: []const u8,
    usage: []const u8,
    examples: []const []const u8 = &.{},
};
