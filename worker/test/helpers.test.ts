import { describe, expect, it } from "vitest";

import { parseAllowedUserIds } from "../src/index";

describe("parseAllowedUserIds", () => {
  it("accepts a comma-separated numeric whitelist", () => {
    expect([...parseAllowedUserIds("851987991, 123456789")]).toEqual([
      851987991,
      123456789,
    ]);
  });

  it("ignores usernames and invalid values", () => {
    expect([...parseAllowedUserIds("ggcaochihieu602, , -1")]).toEqual([]);
  });
});
