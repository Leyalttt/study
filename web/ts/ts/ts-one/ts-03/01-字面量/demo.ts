let direction: "left" | "right" | "up" | "down";

direction = "left";   // 合法
direction = "right";  // 合法
direction = "up";     // 合法
direction = "down";   // 合法
direction = "diagonal"; // 报错：Type '"diagonal"' is not assignable to type '"left" | "right" | "up" | "down"'
// 字面量类型：字面量类型的值是固定的，字面量类型只能取固定的值，不能取其他的值