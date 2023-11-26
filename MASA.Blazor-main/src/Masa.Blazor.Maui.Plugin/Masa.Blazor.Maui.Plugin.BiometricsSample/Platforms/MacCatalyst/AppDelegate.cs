﻿using Foundation;

namespace Masa.Blazor.Maui.Plugin.BiometricsSample
{
    [Register("AppDelegate")]
    public class AppDelegate : MauiUIApplicationDelegate
    {
        protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();
    }
}