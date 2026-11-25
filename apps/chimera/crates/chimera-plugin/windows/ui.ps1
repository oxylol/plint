# The screen, REAPER's windows and dialogs, and the mouse, for ../host-test/run.sh on Windows.
#   ui.ps1 screen | shot <file.jpg> | bounds | click <x> <y> | drag <x1> <y1> <x2> <y2>
#          | dialogs | nag | front | windows
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class W {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc f, IntPtr l);
  [DllImport("user32.dll")] static extern bool EnumChildWindows(IntPtr p, EnumProc f, IntPtr l);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsWindowEnabled(IntPtr h);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, int x, int y, uint d, UIntPtr e);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();

  public static List<IntPtr> Tops(uint pid) {
    var found = new List<IntPtr>();
    EnumWindows((h, l) => {
      uint p;
      GetWindowThreadProcessId(h, out p);
      if (p == pid && IsWindowVisible(h)) found.Add(h);
      return true;
    }, IntPtr.Zero);
    return found;
  }
  public static List<IntPtr> Children(IntPtr parent) {
    var found = new List<IntPtr>();
    EnumChildWindows(parent, (h, l) => { found.Add(h); return true; }, IntPtr.Zero);
    return found;
  }
  public static string Text(IntPtr h) { var s = new StringBuilder(512); GetWindowText(h, s, 512); return s.ToString(); }
  public static string Class(IntPtr h) { var s = new StringBuilder(256); GetClassName(h, s, 256); return s.ToString(); }
}
"@
[W]::SetProcessDPIAware() | Out-Null

$command = $args[0]
$reaper = Get-Process reaper -ErrorAction SilentlyContinue | Select-Object -First 1
function Tops { if ($reaper) { [W]::Tops([uint32]$reaper.Id) } else { @() } }
function Editor { foreach ($h in Tops) { if ([W]::Text($h) -like '*CHIMERA*') { return $h } }; return [IntPtr]::Zero }
function Move($x, $y) { [W]::SetCursorPos($x, $y) | Out-Null; [W]::mouse_event(0x0001, 0, 0, 0, [UIntPtr]::Zero) }

switch ($command) {
  'screen' {
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    "$($b.Width) x $($b.Height)"
  }
  'shot' {
    $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $bitmap = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bitmap.Size)
    $bitmap.Save($args[1], [System.Drawing.Imaging.ImageFormat]::Jpeg)
  }
  'bounds' {
    $h = Editor
    if ($h -ne [IntPtr]::Zero) {
      $r = New-Object W+RECT
      [W]::GetClientRect($h, [ref]$r) | Out-Null
      $p = New-Object W+POINT
      [W]::ClientToScreen($h, [ref]$p) | Out-Null
      "$($p.X) $($p.Y) $($r.R) $($r.B)"
    }
  }
  'click' {
    Move ([int]$args[1]) ([int]$args[2])
    [W]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 50
    [W]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
  }
  'drag' {
    $x1 = [int]$args[1]; $y1 = [int]$args[2]; $x2 = [int]$args[3]; $y2 = [int]$args[4]
    Move $x1 $y1
    Start-Sleep -Milliseconds 200
    [W]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    for ($i = 1; $i -le 10; $i++) {
      Start-Sleep -Milliseconds 60
      Move ([int]($x1 + ($x2 - $x1) * $i / 10)) ([int]($y1 + ($y2 - $y1) * $i / 10))
    }
    Start-Sleep -Milliseconds 200
    [W]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
  }
  'front' {
    $h = Editor
    if ($h -ne [IntPtr]::Zero) { [W]::SetForegroundWindow($h) | Out-Null }
  }
  # REAPER's questions and its nag: answers one, and says what it asked
  'dialogs' {
    $answers = @('No', 'Still Evaluating', 'Still evaluating', 'I agree', 'OK')
    foreach ($top in Tops) {
      if ([W]::Class($top) -ne '#32770') { continue }
      $buttons = @{}
      $said = @()
      foreach ($c in [W]::Children($top)) {
        $text = [W]::Text($c).Replace('&', '')
        switch ([W]::Class($c)) {
          'Button' { if ([W]::IsWindowEnabled($c)) { $buttons[$text] = $c } }
          'Static' { if ($text) { $said += $text } }
        }
      }
      foreach ($a in $answers) {
        if ($buttons.ContainsKey($a)) {
          [W]::SendMessage($buttons[$a], 0x00F5, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
          "$a to: $([W]::Text($top)): $($said -join ' ')"
          return
        }
      }
    }
  }
  'nag' {
    foreach ($h in Tops) { if ([W]::Text($h) -like 'About REAPER*') { 'open'; return } }
  }
  'windows' {
    foreach ($h in Tops) { "$([W]::Class($h)) | $([W]::Text($h))" }
  }
}
